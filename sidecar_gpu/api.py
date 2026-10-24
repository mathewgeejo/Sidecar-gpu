import asyncio
from contextlib import asynccontextmanager
import hashlib
import hmac
from pathlib import Path
import re
import os
from fastapi import APIRouter, Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect
from starlette.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from filelock import FileLock, Timeout
from .arrays import inspect_object
from .config import Config, load_token
from .errors import SidecarError
from .guard import ApiGuard
from .schemas import MatrixJob, EmbeddingJob
from .store import Store, identifier
from .supervisor import Supervisor


def create_app(config=None, token=None, *, start_executor=True):
    config = config or Config()
    store = Store(config)
    token = token or load_token(config)[0]
    supervisor = Supervisor(config, store)
    lock = FileLock(str(store.root / "worker.lock"), timeout=0)

    @asynccontextmanager
    async def lifespan(app):
        try:
            lock.acquire()
        except Timeout as exc:
            raise RuntimeError("another worker owns this data directory") from exc
        try:
            store.initialize()
            if start_executor:
                await asyncio.to_thread(supervisor.start)
            yield
        finally:
            if start_executor:
                await asyncio.to_thread(supervisor.close)
            lock.release()

    app = FastAPI(title="Sidecar GPU", version="0.1.0", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(ApiGuard, token=token)
    app.state.store, app.state.supervisor = store, supervisor

    @app.exception_handler(SidecarError)
    async def sidecar_error(request, exc):
        return JSONResponse(exc.payload(), status_code=exc.status, headers={"Cache-Control": "no-store"})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Never echo an entire request body or authentication header.
        details = [{"location": list(e["loc"]), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse(SidecarError("validation_error", "invalid request", 422, {"fields": details}).payload(), status_code=422)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse(SidecarError("http_error", str(exc.detail), exc.status_code).payload(), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        return JSONResponse(SidecarError("internal_error", "worker internal error; inspect worker console", 500).payload(), status_code=500)

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        return response

    def authenticate(authorization: str | None = Header(default=None)):
        if not authorization or not hmac.compare_digest(authorization.encode("utf-8"), ("Bearer " + token).encode("utf-8")):
            raise SidecarError("unauthorized", "valid bearer token required", 401)

    def require_ready():
        if start_executor and not supervisor.ready:
            raise SidecarError("executor_unavailable", supervisor.error or "executor is starting", 503)

    @app.get("/api/v1/health")
    def health():
        return {"status": "ready" if supervisor.ready else "unavailable", "version": "0.1.0"}

    router = APIRouter(prefix="/api/v1", dependencies=[Depends(authenticate)])

    @router.get("/capabilities")
    def capabilities():
        operations = ["matrix.multiply"] + (["image.embed"] if config.enable_embeddings else [])
        from .models import weights_path, MODEL_ID
        return {"operations": operations, "ready": supervisor.ready,
                "models": [{"id": MODEL_ID, "enabled": config.enable_embeddings, "weights_present": weights_path(config).is_file()}],
                "executor_error": supervisor.error, "device": {**supervisor.info, **store.runtime()},
                "limits": {k: v for k, v in config.to_dict().items() if k != "data_dir"}}

    @router.get("/metrics")
    def metrics():
        return {**store.metrics(), "device": store.runtime(), "executor_ready": supervisor.ready}

    @router.get("/objects/lookup/{digest}")
    def lookup(digest: str):
        obj = store.lookup(digest)
        return {"exists": obj is not None, "object": obj}

    @router.post("/objects", status_code=201)
    async def upload(request: Request, x_content_sha256: str = Header()):
        if not re.fullmatch(r"[0-9a-f]{64}", x_content_sha256):
            raise SidecarError("invalid_hash", "X-Content-SHA256 must be lowercase SHA-256 hex", 422)
        length_header = request.headers.get("content-length")
        if length_header is not None:
            try:
                length = int(length_header)
            except ValueError as exc:
                raise SidecarError("invalid_length", "invalid Content-Length", 422) from exc
            if length <= 0 or length > config.upload_limit:
                raise SidecarError("upload_limit", "upload exceeds configured limit or is empty", 413)
        else:
            length = config.upload_limit
        key = identifier("u_")
        path = store.temp / (key + ".part")
        await run_in_threadpool(store.reserve, key, length)
        count, digest = 0, hashlib.sha256()
        def write_chunk(stream, chunk):
            digest.update(chunk)
            stream.write(chunk)
        try:
            with path.open("xb") as stream:
                async for chunk in request.stream():
                    count += len(chunk)
                    if count > min(length, config.upload_limit):
                        raise SidecarError("upload_limit", "upload exceeds declared or configured limit", 413)
                    await run_in_threadpool(write_chunk, stream, chunk)
                await run_in_threadpool(stream.flush)
                await run_in_threadpool(os.fsync, stream.fileno())
            if length_header is not None and count != length:
                raise SidecarError("incomplete_upload", "body length differs from Content-Length", 422)
            if digest.hexdigest() != x_content_sha256:
                raise SidecarError("hash_mismatch", "uploaded bytes do not match declared SHA-256", 422)
            metadata = await run_in_threadpool(inspect_object, path, config)
            obj = await run_in_threadpool(store.commit_upload, path, digest.hexdigest(), metadata, key)
            return {**obj, "received_bytes": count}
        except ClientDisconnect as exc:
            raise SidecarError("incomplete_upload", "upload disconnected before commit", 400) from exc
        finally:
            path.unlink(missing_ok=True)
            await run_in_threadpool(store.release, key)

    @router.get("/objects/{handle}")
    def object_metadata(handle: str):
        return store.object(handle)

    def download_response(handle):
        path, lease = store.lease(handle)
        return FileResponse(path, media_type="application/octet-stream", filename=handle + ".npy",
                            background=BackgroundTask(store.release_lease, lease))

    @router.get("/objects/{handle}/download")
    def object_download(handle: str):
        return download_response(handle)

    @router.post("/jobs", status_code=202)
    def submit(body: MatrixJob | EmbeddingJob, idempotency_key: str = Header()):
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", idempotency_key):
            raise SidecarError("invalid_idempotency_key", "use 1–128 letters, digits, underscore, dot, colon or dash", 422)
        require_ready()
        job, reused = store.submit(body.model_dump(), idempotency_key)
        return {**job, "idempotency_reused": reused}

    @router.get("/jobs")
    def jobs(limit: int = 50):
        if not 1 <= limit <= 200:
            raise SidecarError("invalid_limit", "limit must be 1–200", 422)
        return {"jobs": store.jobs(limit)}

    @router.get("/jobs/by-key/{key}")
    def by_key(key: str):
        return store.job(idem=key)

    @router.get("/jobs/{job_id}")
    def job_status(job_id: str):
        return store.job(job_id)

    @router.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return store.cancel(job_id)

    def completed_result(job_id):
        job = store.job(job_id)
        if job["state"] != "succeeded":
            raise SidecarError("result_not_ready", "job has no successful result", 409, {"state": job["state"]})
        return store.object(job["result"]["handle"])

    @router.get("/jobs/{job_id}/result")
    def result(job_id: str):
        return completed_result(job_id)

    @router.get("/jobs/{job_id}/result/download")
    def result_download(job_id: str):
        return download_response(completed_result(job_id)["handle"])

    app.include_router(router)
    static = Path(__file__).parent / "static"
    if static.exists():
        @app.get("/dashboard.js", include_in_schema=False)
        def dashboard_script():
            # This TypeScript source uses only browser-executable syntax.
            return FileResponse(static / "dashboard.ts", media_type="text/javascript")
        app.mount("/", StaticFiles(directory=static, html=True), name="dashboard")
    return app
