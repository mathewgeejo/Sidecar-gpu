"""Independent SDK client process for the network lab; never imports PyTorch."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import sys
import time

import httpx
import numpy as np
from sidecar_gpu.client import Client, SubmissionUncertain
from sidecar_gpu.errors import SidecarError


def execute(request):
    token = Path(request["token_file"]).read_text().strip()
    mode = request["mode"]
    result = {"mode": mode, "pid": os.getpid()}
    with Client(request["address"], token, timeout=request.get("timeout", 30),
                receipt_dir=request["receipt_dir"]) as client:
        if mode == "auth":
            with httpx.Client(trust_env=False) as http:
                missing = http.get(request["address"] + "/api/v1/capabilities")
                wrong = http.get(request["address"] + "/api/v1/capabilities",
                                 headers={"Authorization": "Bearer invalid-token"})
            assert missing.status_code == wrong.status_code == 401
            capabilities = client.capabilities()
            assert capabilities["ready"] and capabilities["device"]["device"] == "cpu"
            result.update(missing_status=missing.status_code, wrong_status=wrong.status_code,
                          authenticated_device=capabilities["device"])
        elif mode == "health":
            samples = []
            for _ in range(5):
                started = time.perf_counter()
                response = client.http.get(request["address"] + "/api/v1/health")
                response.raise_for_status()
                assert response.json()["status"] == "ready"
                samples.append(time.perf_counter() - started)
            result.update(samples_s=samples, median_s=statistics.median(samples))
        elif mode in ("prepare", "submit", "upload"):
            paths = [request["a_path"]] if mode == "upload" else [request["a_path"], request["b_path"]]
            objects = [client.upload(path) for path in paths]
            result.update(objects=[asdict(obj) for obj in objects], transfers=list(client.transfers))
            if mode == "submit":
                job = client.multiply(*objects, idempotency_key=request["key"])
                result.update(job_id=job.id, state_at_acceptance=job.state, parameters=job.data["parameters"])
        elif mode == "finish":
            job = client.attach(request["job_id"]).wait(timeout=30)
            transfer = job.download(request["destination"])
            actual = np.load(request["destination"], allow_pickle=False)
            a = np.load(request["a_path"], allow_pickle=False)
            b = np.load(request["b_path"], allow_pickle=False)
            started = time.perf_counter()
            expected = a @ b
            reference_s = time.perf_counter() - started
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
            metadata = job.data["result"]["metadata"]
            assert metadata["execution_device"] == "cpu"
            result.update(job_id=job.id, state=job.state, correct=True, shape=list(actual.shape),
                          execution_device=metadata["execution_device"], timings=job.timings,
                          queue_recovered=job.data["queue_recovered"], download=transfer,
                          local_reference_s=reference_s, max_absolute_error=float(np.max(np.abs(actual - expected))))
        elif mode in ("unreachable", "read_timeout"):
            started = time.perf_counter()
            try:
                client.capabilities()
            except httpx.TransportError as error:
                if mode == "read_timeout":
                    assert isinstance(error, httpx.ReadTimeout), type(error).__name__
                result.update(expected_error=type(error).__name__, elapsed_s=time.perf_counter() - started)
            else:
                raise AssertionError("network fault did not reach the client")
        elif mode == "wait_timeout":
            job = client.attach(request["job_id"])
            try:
                job.wait(timeout=0)
            except TimeoutError:
                result.update(job_id=job.id, expected_error="TimeoutError", state=job.refresh().state)
                assert result["state"] == "queued"
            else:
                raise AssertionError("paused executor unexpectedly completed the job")
        elif mode == "uncertain":
            try:
                client.submit(**request["parameters"], idempotency_key=request["key"])
            except SubmissionUncertain as error:
                receipt = json.loads(error.receipt.read_text())
                assert error.idempotency_key == receipt["idempotency_key"] == request["key"]
                assert token not in error.receipt.read_text()
                result.update(expected_error="SubmissionUncertain", receipt=str(error.receipt),
                              key=error.idempotency_key, receipt_job_id=receipt["job_id"])
            else:
                raise AssertionError("blocked response unexpectedly reached the client")
        elif mode == "recover":
            found = client.find_submission(request["key"])
            repeated = client.submit(**request["parameters"], idempotency_key=request["key"])
            assert found.id == repeated.id
            result.update(job_id=found.id, repeated_job_id=repeated.id,
                          idempotency_reused=repeated.data["idempotency_reused"])
        elif mode in ("broken_upload", "broken_download"):
            destination = Path(request["destination"])
            sentinel = b"previous complete result must survive a failed download"
            if mode == "broken_download":
                destination.write_bytes(sentinel)
            try:
                if mode == "broken_upload":
                    client.upload(request["a_path"])
                else:
                    client.attach(request["job_id"]).download(destination)
            except (httpx.TransportError, SidecarError) as error:
                result.update(expected_error=type(error).__name__)
            else:
                raise AssertionError("partial transfer unexpectedly succeeded")
            if mode == "broken_download":
                assert destination.read_bytes() == sentinel
                assert not list(destination.parent.glob(destination.name + ".*.part"))
                result.update(previous_file_preserved=True, partial_files_removed=True)
        else:
            raise ValueError("unknown client action: " + mode)
    result["torch_imported"] = "torch" in sys.modules
    assert not result["torch_imported"], "client loaded worker-only PyTorch"
    return result


def main():
    request_path, output_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    result = execute(request)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
