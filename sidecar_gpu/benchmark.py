import json
from pathlib import Path
import platform
import tempfile
import time
import uuid
import numpy as np
from .client import Client


def run_benchmark(address, token, *, size=512, repeats=2, require_cuda=False, output=None):
    if not 1 <= size <= 4096 or not 1 <= repeats <= 20:
        raise ValueError("size must be 1–4096 and repeats 1–20")
    rng = np.random.default_rng(12345)
    a = (rng.standard_normal((size, size)) / np.sqrt(size)).astype(np.float32)
    b = (rng.standard_normal((size, size)) / np.sqrt(size)).astype(np.float32)
    with Client(address, token) as client:
        capabilities = client.capabilities()
    if require_cuda and capabilities["device"].get("device") != "cuda":
        raise RuntimeError("demo requires CUDA; worker reports a different execution device")
    report = {"client": {"hostname": platform.node(), "platform": platform.platform()}, "worker_address": address,
              "worker": capabilities["device"], "size": size, "runs": [],
              "verification_level": "CUDA worker over HTTP" if require_cuda else "HTTP worker; inspect execution_device",
              "model_warm": "not applicable: matrix.multiply does not load a model"}
    with tempfile.TemporaryDirectory(prefix="sidecar-benchmark-") as directory:
        directory = Path(directory)
        for index in range(repeats):
            tick = time.perf_counter()
            with Client(address, token) as client:
                ra, rb = client.upload_array(a), client.upload_array(b)
                job = client.multiply(ra, rb, idempotency_key="benchmark-" + uuid.uuid4().hex)
                job_id = job.id
                upload_transfers = list(client.transfers)
            # Closing the submitting HTTP client does not cancel accepted work.
            with Client(address, token) as reconnected:
                job = reconnected.attach(job_id).wait(timeout=180)
                result_path = directory / f"result-{index}.npy"
                download = job.download(result_path)
                end_to_end = time.perf_counter() - tick
                actual = np.load(result_path, allow_pickle=False)
                reference_tick = time.perf_counter()
                expected = a @ b
                reference_s = time.perf_counter() - reference_tick
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
                execution_device = job.data["result"]["metadata"]["execution_device"]
                if require_cuda and execution_device != "cuda":
                    raise AssertionError("result was not executed on CUDA")
                report["runs"].append({"label": "initial_request" if index == 0 else "reused_inputs", "job_id": job_id,
                    "state": job.state, "execution_device": execution_device, "correct": True,
                    "max_absolute_error": float(np.max(np.abs(actual - expected))), "rtol": 1e-4, "atol": 1e-4,
                    "client_end_to_end_s": end_to_end, "local_reference_s": reference_s,
                    "uploaded_payload_bytes": ra.uploaded_bytes + rb.uploaded_bytes,
                    "transfers": upload_transfers + [download], "worker_timings": job.timings,
                    "disconnected_and_reattached": True})
        if repeats > 1 and report["runs"][1]["uploaded_payload_bytes"] != 0:
            raise AssertionError("second run unexpectedly resent inputs")
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
