"""Optional, fixed encoder. Jobs never download weights or accept model files."""
from pathlib import Path
import warnings
from .arrays import file_hash
from .errors import SidecarError

MODEL_ID = "mobilenet_v3_small.imagenet1k_v1.features576"
WEIGHTS_NAME = "mobilenet_v3_small-047dcff4.pth"
WEIGHTS_URL = "https://download.pytorch.org/models/" + WEIGHTS_NAME


def weights_path(config):
    return Path(config.data_dir) / "weights" / WEIGHTS_NAME


def prepare_weights(config):
    import torch
    from filelock import FileLock
    path = weights_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        if not path.exists():
            temporary = path.with_suffix(".download")
            try:
                torch.hub.download_url_to_file(WEIGHTS_URL, temporary, hash_prefix="047dcff4", progress=True)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if not file_hash(path).startswith("047dcff4"):
            raise SidecarError("weights_corrupt", "encoder weights do not match the published checksum", 422)
    return path


def inspect_image(path, config):
    try:
        from PIL import Image
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                width, height = image.size
                if image.format not in ("JPEG", "PNG") or getattr(image, "n_frames", 1) != 1:
                    raise ValueError("only single-frame JPEG and PNG images are supported")
                if max(width, height) > config.max_image_dimension or width * height > config.max_image_pixels:
                    raise ValueError("image dimensions or decoded pixel count exceed limits")
                metadata = {"kind": "image", "format": image.format.lower(), "width": width, "height": height, "mode": image.mode}
                image.verify()
                return metadata
    except ImportError as exc:
        raise SidecarError("image_dependencies_missing", "install the embeddings extra before enabling image uploads", 503) from exc
    except Exception as exc:
        raise SidecarError("invalid_image", str(exc)[:300], 422) from exc


def load_encoder(config, device, torch):
    path = weights_path(config)
    if not path.exists():
        raise SidecarError("weights_unavailable", "run sidecar prepare-model with this worker's data directory before using image.embed", 503)
    if not file_hash(path).startswith("047dcff4"):
        raise SidecarError("weights_corrupt", "encoder weight checksum failed; remove it and run prepare-model", 422)
    try:
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    except ImportError as exc:
        raise SidecarError("model_dependencies_missing", "install matching torchvision and Pillow versions", 503) from exc
    model = mobilenet_v3_small(weights=None)
    # Only the fixed, checksum-verified setup artifact is loaded. Uploads never reach torch.load.
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.classifier = torch.nn.Identity()
    model = model.eval().to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    return model, MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()


def preprocess_images(store, objects, transform, torch):
    from PIL import Image, ImageOps
    tensors = []
    for obj in objects:
        path = store.path(obj["sha256"])
        inspect_image(path, store.config)
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensors.append(transform(image))
    return torch.stack(tensors)
