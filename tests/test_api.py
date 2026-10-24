import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
import pytest
from sidecar_gpu.errors import SidecarError
from sidecar_gpu.schemas import MatrixJob


def npy(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def upload(client, array):
    content = npy(array)
    response = client.post("/api/v1/objects", content=content, headers={"X-Content-SHA256": hashlib.sha256(content).hexdigest()})
    assert response.status_code == 201, response.text
    return response.json()


def inputs(client):
    a = upload(client, np.arange(12, dtype=np.float32).reshape(3, 4))
    b = upload(client, np.ones((4, 2), dtype=np.float32))
    return a, b, {"operation": "matrix.multiply", "a": a["handle"], "b": b["handle"], "a_shape": [3, 4], "b_shape": [4, 2], "dtype": "float32"}


def submit(client, payload, key="test-key"):
    return client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": key})


@pytest.mark.parametrize("authorization", [None, "Bearer wrong"])
def test_authentication(api, authorization):
    client, _ = api
    client.headers.pop("Authorization")
    if authorization:
        client.headers["Authorization"] = authorization
    assert client.get("/api/v1/health").status_code == 200
    for path in ("capabilities", "metrics", "jobs", "objects/o_" + "a" * 32):
        response = client.get("/api/v1/" + path)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"
    assert client.post("/api/v1/objects", content=b"bad").status_code == 401
    assert client.post("/api/v1/jobs", json={}).status_code == 401


def test_hash_dedup_download(api):
    client, store = api
    array = np.ones((4, 4), dtype=np.float32)
    a, b = upload(client, array), upload(client, array)
    assert a["handle"] == b["handle"] and b["cache_hit"]
    lookup = client.get("/api/v1/objects/lookup/" + a["sha256"]).json()
    assert lookup["exists"] and lookup["object"]["handle"] == a["handle"]
    result = client.get("/api/v1/objects/" + a["handle"] + "/download")
    assert np.array_equal(np.load(io.BytesIO(result.content), allow_pickle=False), array)
    bad = client.post("/api/v1/objects", content=npy(array), headers={"X-Content-SHA256": "0" * 64})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "hash_mismatch"
    assert store.metrics()["object_count"] == 1
    assert store.metrics()["reserved_bytes"] == 0


@pytest.mark.parametrize("array", [np.ones((2, 2), dtype=np.float64), np.ones((2,), dtype=np.float32),
                                      np.ones((0, 2), dtype=np.float32), np.ones((2, 2), dtype=np.int32),
                                      np.asfortranarray(np.ones((2, 3), dtype=np.float32))])
def test_unsupported_arrays(api, array):
    client, _ = api
    content = npy(array)
    response = client.post("/api/v1/objects", content=content, headers={"X-Content-SHA256": hashlib.sha256(content).hexdigest()})
    assert response.status_code == 422


@pytest.mark.parametrize("kind", ["object", "truncated", "trailing", "pickle", "huge_header"])
def test_malicious_arrays(api, kind):
    client, store = api
    if kind == "object":
        stream = io.BytesIO()
        np.save(stream, np.array([[{}]], dtype=object), allow_pickle=True)
        content = stream.getvalue()
    elif kind == "pickle":
        content = b"\x80\x04arbitrary serialization"
    elif kind == "huge_header":
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {"shape": (999999999, 999999999), "fortran_order": False, "descr": "<f4"})
        content = stream.getvalue()
    else:
        content = npy(np.ones((3, 3), dtype=np.float32))
        content = content[:-4] if kind == "truncated" else content + b"extra"
    response = client.post("/api/v1/objects", content=content, headers={"X-Content-SHA256": hashlib.sha256(content).hexdigest()})
    assert response.status_code == 422
    assert store.metrics()["object_count"] == 0
    assert not list(store.temp.iterdir())


def test_job_idempotency_validation_cancel(api):
    client, store = api
    _, _, payload = inputs(client)
    first = submit(client, payload)
    assert first.status_code == 202, first.text
    job = first.json()
    assert submit(client, payload).json()["id"] == job["id"]
    assert submit(client, {**payload, "retain_output": False}).status_code == 409
    assert submit(client, {**payload, "a_shape": [2, 6]}, "bad-shape").status_code == 422
    assert submit(client, {**payload, "dtype": "float64"}, "bad-type").status_code == 422
    assert submit(client, {**payload, "operation": "os.execute"}, "bad-op").status_code == 422
    assert client.get("/api/v1/jobs/by-key/test-key").json()["id"] == job["id"]
    assert client.get(f"/api/v1/jobs/{job['id']}/result").status_code == 409
    assert client.post(f"/api/v1/jobs/{job['id']}/cancel").json()["state"] == "cancelled"
    assert store.claim() is None
    assert store.metrics()["reserved_bytes"] == 0


def test_inner_dimensions(api):
    client, _ = api
    a = upload(client, np.ones((2, 3), dtype=np.float32))
    b = upload(client, np.ones((4, 2), dtype=np.float32))
    response = submit(client, {"a": a["handle"], "b": b["handle"], "a_shape": [2, 3], "b_shape": [4, 2]})
    assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_dimensions"


def test_upload_queue_output_limits(api):
    client, store = api
    store.config.upload_limit = 150
    content = npy(np.ones((10, 10), dtype=np.float32))
    assert client.post("/api/v1/objects", content=content, headers={"X-Content-SHA256": hashlib.sha256(content).hexdigest()}).status_code == 413
    store.config.upload_limit = 128 * 1024**2
    _, _, payload = inputs(client)
    store.config.output_limit = 4
    assert submit(client, payload).status_code == 413
    store.config.output_limit = 128 * 1024**2
    store.config.queue_limit = 1
    assert submit(client, payload).status_code == 202
    assert submit(client, payload, "second").status_code == 429
    assert submit(client, payload).status_code == 202  # retry accepted even when full


def test_restart_reconciliation_and_cancellation_boundary(api):
    client, store = api
    _, _, payload = inputs(client)
    running = submit(client, payload, "running").json()
    queued = submit(client, payload, "queued").json()
    assert store.claim()["id"] == running["id"]
    assert store.cancel(running["id"])["state"] == "cancel_requested"
    store.reconcile(startup=True)
    assert store.job(running["id"])["state"] == "interrupted"
    assert store.job(queued["id"])["state"] == "queued"
    assert store.claim()["id"] == queued["id"]
    store.cancel(queued["id"])
    # Result acceptance races with cancellation; cancellation wins before commit.
    store.finish(queued["id"], "failed", {}, error={"code": "example", "message": "example"})
    assert store.job(queued["id"])["state"] == "cancelled"


def test_recovery_rejects_corrupt_input(api):
    client, store = api
    a, _, payload = inputs(client)
    job = submit(client, payload).json()
    path = store.path(a["sha256"])
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    store.reconcile(startup=True)
    assert store.job(job["id"])["state"] == "failed"


def test_eviction_protects_inputs_and_downloads(api):
    client, store = api
    a, b, payload = inputs(client)
    job = submit(client, payload).json()
    used = store.metrics()["storage_bytes"]
    reserved = store.metrics()["reserved_bytes"]
    store.config.disk_budget = used + reserved
    with pytest.raises(SidecarError, match="disk budget"):
        store.reserve("too-big", 1)
    assert store.object(a["handle"]) and store.object(b["handle"])
    store.cancel(job["id"])
    _, lease = store.lease(a["handle"])
    store.config.disk_budget = used
    store.reserve("evict-one", b["bytes"])
    assert store.object(a["handle"])
    assert client.get("/api/v1/objects/" + b["handle"]).status_code == 410
    store.release_lease(lease)
    store.release("evict-one")


def test_results_and_retention(api):
    client, store = api
    _, _, payload = inputs(client)
    job = submit(client, payload).json()
    store.claim()
    path = store.temp / "output.npy"
    array = np.arange(12, dtype=np.float32).reshape(3, 4) @ np.ones((4, 2), dtype=np.float32)
    np.save(path, array, allow_pickle=False)
    from sidecar_gpu.arrays import inspect_array
    meta = {**inspect_array(path, store.config, output=True), "execution_device": "cpu"}
    store.finish(job["id"], "succeeded", {"compute_s": 0.1}, temporary=path, metadata=meta)
    response = client.get(f"/api/v1/jobs/{job['id']}/result/download")
    np.testing.assert_array_equal(np.load(io.BytesIO(response.content)), array)
    result = store.job(job["id"])["result"]
    with store.transaction() as db:
        db.execute("UPDATE objects SET protected_until=0 WHERE id=?", (result["handle"],))
    store.config.disk_budget = 4096
    store.reserve("force-eviction", 4096)
    assert client.get(f"/api/v1/jobs/{job['id']}/result").status_code == 410


def test_second_worker_lock(api):
    client, store = api
    from fastapi.testclient import TestClient
    from sidecar_gpu.api import create_app
    with pytest.raises(RuntimeError, match="another worker"):
        with TestClient(create_app(store.config, "x" * 40, start_executor=False)):
            pass
