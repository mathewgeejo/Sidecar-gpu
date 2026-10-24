import os
from pathlib import Path
import time
import numpy as np
import psutil
import pytest
from sidecar_gpu.client import Client
from sidecar_gpu.benchmark import run_benchmark


@pytest.mark.integration
def test_http_execution_reconnect_cache_and_timings(http_worker, tmp_path):
    address, token, _ = http_worker
    report = run_benchmark(address, token, size=128, repeats=2, output=tmp_path / "benchmark.json")
    assert all(run["correct"] for run in report["runs"])
    assert all(run["execution_device"] == "cpu" for run in report["runs"])
    assert report["runs"][0]["uploaded_payload_bytes"] > 0
    assert report["runs"][1]["uploaded_payload_bytes"] == 0
    for run in report["runs"]:
        assert run["worker_timings"]["compute_s"] > 0
        assert run["worker_timings"]["gpu_compute_s"] is None
        assert run["client_end_to_end_s"] >= run["worker_timings"]["compute_s"]


@pytest.mark.integration
def test_executor_crash_replacement(http_worker):
    address, token, _ = http_worker
    with Client(address, token) as client:
        old_pid = client.capabilities()["device"]["executor_pid"]
        psutil.Process(old_pid).kill()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            caps = client.capabilities()
            if caps["ready"] and caps["device"]["executor_pid"] != old_pid:
                break
            time.sleep(0.1)
        else:
            pytest.fail("executor was not replaced")
        a = client.upload_array(np.eye(3, dtype=np.float32))
        b = client.upload_array(np.ones((3, 2), dtype=np.float32))
        job = client.multiply(a, b).wait()
        assert job.state == "succeeded"
        assert job.data["result"]["metadata"]["execution_device"] == "cpu"


@pytest.mark.integration
def test_receipt_and_timeout_preserve_job(http_worker, tmp_path):
    address, token, _ = http_worker
    with Client(address, token, receipt_dir=tmp_path / "receipts") as client:
        a = client.upload_array(np.ones((100, 100), dtype=np.float32))
        job = client.multiply(a, a, idempotency_key="durable-receipt")
        repeated = client.multiply(a, a, idempotency_key="durable-receipt")
        assert repeated.id == job.id
        assert len(list((tmp_path / "receipts").glob("*.json"))) == 1
    with Client(address, token) as reattached:
        assert reattached.find_submission("durable-receipt").wait().state == "succeeded"


@pytest.mark.integration
def test_full_worker_restart_recovers_queue_and_completed_results(tmp_path):
    from tests.conftest import running_worker, TOKEN
    from sidecar_gpu.config import Config
    from sidecar_gpu.schemas import MatrixJob
    from sidecar_gpu.store import Store
    config = Config(data_dir=str(tmp_path / "restart-worker"), device="cpu")
    with running_worker(config) as (address, token, _):
        with Client(address, token) as client:
            a = client.upload_array(np.eye(5, dtype=np.float32))
            b = client.upload_array(np.ones((5, 3), dtype=np.float32))
    # Seed durable states while there is no executor, reproducing an abrupt loss
    # immediately after claim without relying on a timing race in a short kernel.
    store = Store(config)
    payload = MatrixJob(a=a.handle, b=b.handle, a_shape=[5, 5], b_shape=[5, 3]).model_dump()
    old, _ = store.submit(payload, "lost-attempt")
    assert store.claim()["id"] == old["id"]
    queued, _ = store.submit(payload, "recover-queue")
    with running_worker(config) as (address, token, _):
        with Client(address, token) as client:
            assert client.attach(old["id"]).state == "interrupted"
            done = client.attach(queued["id"]).wait()
            done.download(tmp_path / "recovered.npy")
            assert done.data["queue_recovered"]
            np.testing.assert_array_equal(np.load(tmp_path / "recovered.npy"), np.ones((5, 3), dtype=np.float32))
    with running_worker(config) as (address, token, _):
        with Client(address, token) as client:
            completed = client.attach(queued["id"])
            assert completed.state == "succeeded"
            completed.download(tmp_path / "after-restart.npy")
            repeated = client.multiply(a, b, idempotency_key="recover-queue")
            assert repeated.id == queued["id"]
