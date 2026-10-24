from dataclasses import asdict, dataclass, fields
from pathlib import Path
import json
import os


@dataclass
class Config:
    data_dir: str = str(Path.home() / ".sidecar-gpu")
    device: str = "cuda"
    queue_limit: int = 8
    upload_limit: int = 128 * 1024**2
    output_limit: int = 128 * 1024**2
    disk_budget: int = 4 * 1024**3
    max_dimension: int = 8192
    result_retention_seconds: int = 86400
    transient_result_seconds: int = 600
    input_grace_seconds: int = 300
    gpu_memory_fraction: float = 0.75
    host_memory_fraction: float = 0.5
    startup_timeout: float = 60.0
    enable_embeddings: bool = False
    max_images: int = 16
    max_image_dimension: int = 4096
    max_image_pixels: int = 16777216

    def __post_init__(self):
        self.data_dir = str(Path(self.data_dir).expanduser().resolve())
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        for name in ("queue_limit", "upload_limit", "output_limit", "disk_budget", "max_dimension", "max_images", "max_image_dimension", "max_image_pixels"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.gpu_memory_fraction < 1 or not 0 < self.host_memory_fraction < 1:
            raise ValueError("memory fractions must be between zero and one")
        if min(self.result_retention_seconds, self.transient_result_seconds, self.input_grace_seconds) < 0:
            raise ValueError("retention durations cannot be negative")
        if self.max_images > 16:
            raise ValueError("max_images may be reduced but cannot exceed the prototype hard limit of 16")

    @classmethod
    def load(cls, path=None, **overrides):
        data = json.loads(Path(path).read_text()) if path else {}
        valid = {f.name for f in fields(cls)}
        if set(data) - valid:
            raise ValueError(f"unknown configuration keys: {set(data) - valid}")
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def to_dict(self):
        return asdict(self)


def load_token(config, token_file=None):
    """Never return the token in status responses or print it automatically."""
    import secrets
    path = Path(token_file) if token_file else Path(config.data_dir) / "pairing.token"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(secrets.token_urlsafe(32) + "\n")
    token = path.read_text().strip()
    if len(token) < 32:
        raise ValueError("pairing token must contain at least 32 characters")
    return token, path
