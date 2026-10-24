"""Read a bounded NPY header before exposing any array allocation."""
import hashlib
import math
import numpy as np
from .errors import SidecarError


def inspect_array(path, config, *, output=False):
    try:
        with open(path, "rb") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=4096)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=4096)
            else:
                raise ValueError("only NPY versions 1 and 2 are supported")
            offset = stream.tell()
            if len(shape) != 2 or any(type(n) is not int or n <= 0 or n > config.max_dimension for n in shape):
                raise ValueError("expected two positive, bounded dimensions")
            if dtype != np.dtype("<f4") or fortran or dtype.hasobject:
                raise ValueError("only C-contiguous little-endian float32 matrices are supported")
            size = math.prod(shape) * 4
            limit = config.output_limit if output else config.upload_limit
            if size > limit or path.stat().st_size != offset + size:
                raise ValueError("array payload exceeds limit or does not match its header")
        return {"shape": list(shape), "dtype": "float32", "format": "npy", "payload_bytes": size}
    except (ValueError, OSError, EOFError, TypeError, OverflowError) as exc:
        raise SidecarError("invalid_array", str(exc), 422) from exc


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_object(path, config, *, output=False):
    with path.open("rb") as stream:
        magic = stream.read(8)
    if magic.startswith(b"\x93NUMPY"):
        return inspect_array(path, config, output=output)
    if config.enable_embeddings and (magic.startswith(b"\xff\xd8") or magic == b"\x89PNG\r\n\x1a\n"):
        from .models import inspect_image
        return inspect_image(path, config)
    raise SidecarError("invalid_array", "expected safe NPY float32 matrix" + (" or bounded JPEG/PNG image" if config.enable_embeddings else ""), 422)
