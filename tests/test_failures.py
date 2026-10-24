import asyncio
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import io
import numpy as np
import pytest
from sidecar_gpu.client import Client, Job, JobFailed, SubmissionUncertain
from sidecar_gpu.errors import SidecarError
from tests.test_api import inputs, submit


def test_json_body_limit_and_early_auth(api):
    client, _ = api
    large = b" " * 70000
    assert client.post("/api/v1/jobs", content=large).status_code == 413
    client.headers.pop("Authorization")
    assert client.post("/api/v1/jobs", content=large).status_code == 401


def test_model_free_host_memory_admission(api, monkeypatch):
    client, store = api
    _, _, payload = inputs(client)
    job_id = submit(client, payload).json()["id"]
    job = store.claim()
    import torch
    import psutil
    from sidecar_gpu.executor import execute_job
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=1024))
    execute_job(store, job, "cpu", torch)
    failed = store.job(job_id)
    assert failed["state"] == "failed"
    assert failed["error"]["code"] == "host_memory_limit"
    assert store.metrics()["reserved_bytes"] == 0


def test_replacement_keeps_live_upload_reservations(api):
    _, store = api
    store.reserve("u_live", 4096)
    store.reconcile(startup=False)
    assert store.metrics()["reserved_bytes"] == 4096
    store.release("u_live")


def test_submission_timeout_recovers_existing_job(tmp_path):
    import httpx
    calls = []
    def handler(request):
        calls.append(request.method)
        if request.method == "POST":
            raise httpx.ReadTimeout("lost response", request=request)
        return httpx.Response(200, json={"id": "j_" + "a" * 32, "state": "queued", "timings": {}})
    with Client("http://worker", "secret", receipt_dir=tmp_path) as client:
        client.http.close()
        client.http = httpx.Client(base_url="http://worker/api/v1/", transport=httpx.MockTransport(handler))
        job = client.submit("matrix.multiply", a="o_" + "a" * 32, b="o_" + "b" * 32,
                            a_shape=[2, 2], b_shape=[2, 2], idempotency_key="same-key")
        assert job.id == "j_" + "a" * 32
        assert calls == ["POST", "GET"]


def test_uncertain_submission_records_key_and_never_resubmits(tmp_path):
    import httpx
    calls = []
    def handler(request):
        calls.append(request.method)
        raise httpx.ConnectError("worker unreachable", request=request)
    with Client("http://worker", "secret", receipt_dir=tmp_path) as client:
        client.http.close()
        client.http = httpx.Client(base_url="http://worker/api/v1/", transport=httpx.MockTransport(handler))
        with pytest.raises(SubmissionUncertain) as error:
            client.submit("matrix.multiply", a="o_" + "a" * 32, b="o_" + "b" * 32,
                          a_shape=[2, 2], b_shape=[2, 2], idempotency_key="uncertain")
        assert error.value.idempotency_key == "uncertain"
        assert error.value.receipt.exists()
        assert calls == ["POST", "GET"]


def test_wait_timeout_does_not_cancel():
    class FakeClient:
        def __init__(self):
            self.methods = []
        def _json(self, method, path):
            self.methods.append(method)
            return {"id": "test", "state": "running", "timings": {}}
    client = FakeClient()
    job = Job(client, {"id": "test", "state": "queued"})
    with pytest.raises(TimeoutError, match="reattach"):
        job.wait(timeout=0)
    assert client.methods == ["GET"]


def test_cuda_worker_never_falls_back(tmp_path):
    import torch
    if torch.version.cuda is not None:
        pytest.skip("this check targets an explicitly CPU-only PyTorch installation")
    result = subprocess.run([sys.executable, "-m", "sidecar_gpu", "worker", "--device", "cuda", "--data-dir", str(tmp_path / "cuda")],
                            capture_output=True, text=True, timeout=45)
    assert result.returncode != 0
    assert "CUDA unavailable" in result.stderr


def test_chunked_upload_limit_cleans_reservation(api):
    _, store = api
    from sidecar_gpu.api import create_app
    import httpx
    async def run():
        config = store.config
        config.upload_limit = 64
        application = create_app(config, "x" * 40, start_executor=False)
        async def chunks():
            yield b"a" * 40
            yield b"b" * 40
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://test") as client:
            return await client.post("/api/v1/objects", content=chunks(), headers={"Authorization": "Bearer " + "x" * 40, "X-Content-SHA256": "0" * 64})
    response = asyncio.run(run())
    assert response.status_code == 413
    assert store.metrics()["reserved_bytes"] == 0
    assert not list(store.temp.iterdir())
