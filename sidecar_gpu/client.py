"""Synchronous Python SDK; torch is not needed on the client machine."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import quote
import uuid
import httpx
import numpy as np
from .arrays import file_hash
from .errors import SidecarError

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


class JobFailed(SidecarError):
    def __init__(self, data):
        self.job = data
        error = data.get("error") or {"code": data["state"], "message": "job ended: " + data["state"]}
        super().__init__(error["code"], error["message"], 409)


class SubmissionUncertain(Exception):
    def __init__(self, key, receipt):
        self.idempotency_key, self.receipt = key, receipt
        super().__init__(f"submission response unavailable; look up idempotency key {key}; receipt: {receipt}")


@dataclass
class RemoteObject:
    handle: str
    sha256: str
    bytes: int
    metadata: dict
    cache_hit: bool = False
    uploaded_bytes: int = 0
    upload_s: float = 0
    lookup_s: float = 0
    hash_s: float = 0


class Job:
    def __init__(self, client, data):
        self.client, self.data = client, data
        self.id = data["id"]

    @property
    def state(self):
        return self.data["state"]

    @property
    def timings(self):
        return self.data["timings"]

    def refresh(self):
        self.data = self.client._json("GET", "/jobs/" + self.id)
        return self

    def wait(self, timeout=120, poll_interval=0.1):
        deadline = time.monotonic() + timeout
        while True:
            self.refresh()
            if self.state in TERMINAL:
                if self.state != "succeeded":
                    raise JobFailed(self.data)
                return self
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"job {self.id} is still {self.state}; reattach using its ID")
            time.sleep(min(poll_interval, remaining))

    def cancel(self):
        self.data = self.client._json("POST", "/jobs/" + self.id + "/cancel")
        return self

    def result_object(self):
        return self.client._json("GET", "/jobs/" + self.id + "/result")

    def download(self, destination):
        return self.client.download(self.result_object()["handle"], destination)


class Client:
    def __init__(self, address, token, *, timeout=30, receipt_dir=None):
        self.address = address.rstrip("/")
        if not self.address.startswith(("http://", "https://")):
            raise ValueError("address must include http:// or https://")
        self.http = httpx.Client(base_url=self.address + "/api/v1/", headers={"Authorization": "Bearer " + token},
                                 timeout=timeout, trust_env=False, follow_redirects=False)
        self.receipt_dir = Path(receipt_dir or Path.home() / ".sidecar-gpu-client" / "receipts")
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        self.transfers = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.http.close()

    def _json(self, method, path, **kwargs):
        response = self.http.request(method, path.lstrip("/"), **kwargs)
        self._check(response)
        return response.json()

    @staticmethod
    def _check(response):
        if response.is_error:
            try:
                error = response.json()["error"]
            except (ValueError, KeyError):
                error = {"code": "http_error", "message": f"worker returned HTTP {response.status_code}"}
            raise SidecarError(error["code"], error["message"], response.status_code, error.get("details"))

    def capabilities(self):
        return self._json("GET", "/capabilities")

    def upload(self, source):
        path = Path(source)
        tick = time.perf_counter()
        digest, size = file_hash(path), path.stat().st_size
        hash_s = time.perf_counter() - tick
        tick = time.perf_counter()
        lookup = self._json("GET", "/objects/lookup/" + digest)
        lookup_s = time.perf_counter() - tick
        tick = time.perf_counter()
        if lookup["exists"]:
            obj, sent = lookup["object"], 0
            hit = True
        else:
            with path.open("rb") as stream:
                obj = self._json("POST", "/objects", content=stream,
                                 headers={"X-Content-SHA256": digest, "Content-Length": str(size), "Content-Type": "application/octet-stream"})
            sent, hit = size, obj["cache_hit"]
        duration = time.perf_counter() - tick
        transfer = {"direction": "upload", "bytes": sent, "duration_s": duration, "lookup_s": lookup_s,
                    "hash_s": hash_s, "cache_hit": hit, "sha256": digest}
        self.transfers.append(transfer)
        return RemoteObject(obj["handle"], digest, obj["bytes"], obj["metadata"], hit, sent, duration, lookup_s, hash_s)

    def upload_array(self, array):
        if array.dtype != np.float32 or array.ndim != 2:
            raise ValueError("upload_array requires a two-dimensional float32 array")
        with tempfile.TemporaryDirectory(prefix="sidecar-client-") as directory:
            path = Path(directory) / "input.npy"
            np.save(path, np.ascontiguousarray(array), allow_pickle=False)
            return self.upload(path)

    def _receipt(self, key, payload, job_id=None):
        import hashlib
        # Hash the key/address for filenames; caller-supplied keys cannot traverse paths.
        name = hashlib.sha256((self.address + "\0" + key).encode()).hexdigest()
        path = self.receipt_dir / (name + ".json")
        record = {"address": self.address, "idempotency_key": key, "parameters": payload, "job_id": job_id}
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous["parameters"] != payload:
                raise SidecarError("idempotency_conflict", "local receipt already binds this key to different parameters", 409)
            if job_id is None:
                record["job_id"] = previous.get("job_id")
        temporary = self.receipt_dir / (name + "." + uuid.uuid4().hex + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        return path

    def _submit_payload(self, payload, idempotency_key=None):
        key = idempotency_key or str(uuid.uuid4())
        receipt = self._receipt(key, payload)
        try:
            data = self._json("POST", "/jobs", json=payload, headers={"Idempotency-Key": key})
        except httpx.TransportError as exc:
            try:
                data = self._json("GET", "/jobs/by-key/" + quote(key, safe=""))
            except (httpx.TransportError, SidecarError):
                raise SubmissionUncertain(key, receipt) from exc
        self._receipt(key, payload, data["id"])
        return Job(self, data)

    def submit(self, operation, *, a, b, a_shape, b_shape, dtype="float32", retain_output=True, idempotency_key=None, retry_of=None):
        payload = {"operation": operation, "a": a, "b": b, "a_shape": list(a_shape), "b_shape": list(b_shape),
                   "dtype": dtype, "retain_output": retain_output, "retry_of": retry_of}
        return self._submit_payload(payload, idempotency_key)

    def embed(self, images, *, retain_output=True, idempotency_key=None):
        return self._submit_payload({"operation": "image.embed", "images": [image.handle if isinstance(image, RemoteObject) else image for image in images],
                                     "model": "mobilenet_v3_small.imagenet1k_v1.features576", "retain_output": retain_output, "retry_of": None}, idempotency_key)

    def multiply(self, a: RemoteObject, b: RemoteObject, **kwargs):
        return self.submit("matrix.multiply", a=a.handle, b=b.handle, a_shape=a.metadata["shape"], b_shape=b.metadata["shape"], **kwargs)

    def attach(self, job_id):
        return Job(self, self._json("GET", "/jobs/" + quote(job_id, safe="")))

    def find_submission(self, key):
        return Job(self, self._json("GET", "/jobs/by-key/" + quote(key, safe="")))

    def retry(self, job_id, *, idempotency_key=None):
        job = self.attach(job_id)
        params = {**job.data["parameters"], "retry_of": job_id}
        return self._submit_payload(params, idempotency_key)

    def download(self, handle, destination):
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
        tick, count = time.perf_counter(), 0
        try:
            with self.http.stream("GET", "objects/" + quote(handle, safe="") + "/download") as response:
                if response.is_error:
                    response.read()
                    self._check(response)
                with temporary.open("wb") as stream:
                    for chunk in response.iter_bytes(1024 * 1024):
                        stream.write(chunk)
                        count += len(chunk)
            metadata = self._json("GET", "/objects/" + quote(handle, safe=""))
            if count != metadata["bytes"] or file_hash(temporary) != metadata["sha256"]:
                raise SidecarError("download_corrupt", "download did not match verified object hash", 502)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        record = {"direction": "download", "bytes": count, "duration_s": time.perf_counter() - tick}
        self.transfers.append(record)
        return record
