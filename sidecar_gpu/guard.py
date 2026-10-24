"""Authenticate before FastAPI buffers/parses JSON; bound non-upload request bodies."""
import hmac
from starlette.responses import JSONResponse
from .errors import SidecarError


class ApiGuard:
    def __init__(self, app, token):
        self.app = app
        self.expected = ("Bearer " + token).encode("utf-8")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/api/v1/") or scope["path"] == "/api/v1/health":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        if not hmac.compare_digest(headers.get(b"authorization", b""), self.expected):
            response = JSONResponse(SidecarError("unauthorized", "valid bearer token required", 401).payload(), 401,
                                    headers={"Cache-Control": "no-store"})
            return await response(scope, receive, send)
        if scope["method"] in ("POST", "PUT", "PATCH") and scope["path"] != "/api/v1/objects":
            parts, size = [], 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                size += len(message.get("body", b""))
                if size > 65536:
                    response = JSONResponse(SidecarError("request_limit", "JSON request exceeds 64 KiB", 413).payload(), 413)
                    return await response(scope, receive, send)
                parts.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            pending = b"".join(parts)
            consumed = False
            async def replay():
                nonlocal consumed
                if not consumed:
                    consumed = True
                    return {"type": "http.request", "body": pending, "more_body": False}
                return await receive()
            return await self.app(scope, replay, send)
        return await self.app(scope, receive, send)
