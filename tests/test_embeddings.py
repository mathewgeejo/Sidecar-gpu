import hashlib
import io
from pathlib import Path
import numpy as np
import pytest
from sidecar_gpu.client import Client, JobFailed


def image_bytes(size=(40, 30), format="PNG"):
    Image = pytest.importorskip("PIL.Image")
    stream = io.BytesIO()
    Image.new("RGB", size, color=(80, 120, 40)).save(stream, format=format)
    return stream.getvalue()


def send_image(client, content):
    return client.post("/api/v1/objects", content=content, headers={"X-Content-SHA256": hashlib.sha256(content).hexdigest()})


def test_image_limits_and_operation_types(api):
    client, store = api
    assert send_image(client, image_bytes()).status_code == 422  # disabled by default
    store.config.enable_embeddings = True
    response = send_image(client, image_bytes())
    assert response.status_code == 201
    handle = response.json()["handle"]
    store.config.max_image_dimension = 32
    assert send_image(client, image_bytes()).status_code == 422
    store.config.max_image_dimension = 4096
    malformed = b"\x89PNG\r\n\x1a\nnot-an-image"
    assert send_image(client, malformed).status_code == 422
    payload = {"operation": "image.embed", "images": [handle]}
    assert client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": "image"}).status_code == 202
    store.config.max_images = 1
    assert client.post("/api/v1/jobs", json={**payload, "images": [handle, handle]}, headers={"Idempotency-Key": "too-many"}).status_code == 413
    matrix = {"operation": "matrix.multiply", "a": handle, "b": handle, "a_shape": [40, 30], "b_shape": [40, 30]}
    assert client.post("/api/v1/jobs", json=matrix, headers={"Idempotency-Key": "wrong-type"}).status_code == 422


def test_missing_weights_fail_honestly(api):
    client, store = api
    store.config.enable_embeddings = True
    handle = send_image(client, image_bytes()).json()["handle"]
    payload = {"operation": "image.embed", "images": [handle]}
    job_id = client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": "missing-weights"}).json()["id"]
    job = store.claim()
    import torch
    from sidecar_gpu.executor import execute_job
    execute_job(store, job, "cpu", torch, {})
    assert store.job(job_id)["state"] == "failed"
    assert store.job(job_id)["error"]["code"] == "weights_unavailable"


@pytest.mark.integration
def test_encoder_real_http_cold_warm_and_reference(embedding_worker, tmp_path):
    address, token, config = embedding_worker
    path = tmp_path / "input.png"
    path.write_bytes(image_bytes())
    outputs = []
    with Client(address, token) as client:
        image = client.upload(path)
        first = client.embed([image, image]).wait()
        assert first.timings["model_cache_hit"] is False
        assert first.timings["model_load_s"] > 0
        first.download(tmp_path / "first.npy")
        second = client.embed([image, image]).wait()
        assert second.timings["model_cache_hit"] is True
        assert second.timings["model_load_s"] == 0
        second.download(tmp_path / "second.npy")
        assert client.upload(path).uploaded_bytes == 0
    actual = np.load(tmp_path / "first.npy", allow_pickle=False)
    assert actual.shape == (2, 576)
    np.testing.assert_array_equal(actual, np.load(tmp_path / "second.npy", allow_pickle=False))
    # Separate CPU reference after downloading the result; SDK never computes it.
    import torch
    from PIL import Image, ImageOps
    from sidecar_gpu.models import load_encoder
    model, transform = load_encoder(config, "cpu", torch)
    with Image.open(path) as original:
        tensor = transform(ImageOps.exif_transpose(original).convert("RGB"))
    with torch.inference_mode():
        expected = model(torch.stack([tensor, tensor])).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
