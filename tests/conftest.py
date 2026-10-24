import asyncio
import multiprocessing as mp
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import time
import httpx
import pytest
from fastapi.testclient import TestClient
from sidecar_gpu.api import create_app
from sidecar_gpu.config import Config

TOKEN = "test-pairing-token-with-at-least-32-characters"


@pytest.fixture
def config(tmp_path):
    return Config(data_dir=str(tmp_path / "worker"), device="cpu", input_grace_seconds=0)


@pytest.fixture
def api(config):
    application = create_app(config, TOKEN, start_executor=False)
    with TestClient(application) as client:
        client.headers["Authorization"] = "Bearer " + TOKEN
        yield client, application.state.store


def serve(config_dict, port, stop):
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(create_app(Config(**config_dict), TOKEN), host="127.0.0.1", port=port,
                                         access_log=False, log_level="warning"))
    async def run():
        async def watch():
            while not stop.is_set():
                await asyncio.sleep(0.1)
            server.should_exit = True
        watcher = asyncio.create_task(watch())
        try:
            await server.serve()
        finally:
            watcher.cancel()
    asyncio.run(run())


@pytest.fixture(scope="module")
def http_worker(tmp_path_factory):
    directory = tmp_path_factory.mktemp("http-worker")
    settings = Config(data_dir=str(directory), device="cpu")
    with running_worker(settings) as worker:
        yield worker


@contextmanager
def running_worker(settings):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    stop = context.Event()
    process = context.Process(target=serve, args=(settings.to_dict(), port, stop))
    process.start()
    address = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 90
    try:
        with httpx.Client(trust_env=False, timeout=1) as client:
            while time.monotonic() < deadline:
                if not process.is_alive():
                    pytest.fail("worker exited during startup")
                try:
                    if client.get(address + "/api/v1/health").json()["status"] == "ready":
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.1)
            else:
                pytest.fail("HTTP worker failed to start")
        yield address, TOKEN, settings
    finally:
        stop.set()
        process.join(20)
        if process.is_alive():
            process.terminate()
            process.join(10)


@pytest.fixture(scope="module")
def embedding_worker(tmp_path_factory):
    import shutil
    source = os.environ.get("SIDECAR_TEST_WEIGHTS")
    if not source or not Path(source).is_file():
        pytest.skip("set SIDECAR_TEST_WEIGHTS to the prepare-model checkpoint for offline model tests")
    pytest.importorskip("torchvision")
    from sidecar_gpu.models import weights_path
    settings = Config(data_dir=str(tmp_path_factory.mktemp("embedding-worker")), device="cpu", enable_embeddings=True)
    destination = weights_path(settings)
    destination.parent.mkdir(parents=True)
    shutil.copyfile(source, destination)
    with running_worker(settings) as worker:
        yield worker
