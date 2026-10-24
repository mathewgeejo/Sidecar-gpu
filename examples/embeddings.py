"""Demonstrate cold/warm model loading and reuse of image objects over HTTP."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from sidecar_gpu.client import Client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="http://127.0.0.1:8377")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--images", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("embedding-benchmark.json"))
    args = parser.parse_args()
    token = os.environ.get("SIDECAR_TOKEN") or args.token_file.read_text().strip()
    records = []
    with tempfile.TemporaryDirectory(prefix="sidecar-images-") as directory:
        directory = Path(directory)
        paths = args.images
        if not paths:
            from PIL import Image, ImageDraw
            paths = []
            for index, color in enumerate(((65, 112, 174), (219, 122, 48))):
                image = Image.new("RGB", (320, 240), color=(240, 242, 236))
                ImageDraw.Draw(image).rectangle((70, 40, 240, 210), fill=color)
                path = directory / f"sample-{index}.png"
                image.save(path)
                paths.append(path)
        with Client(args.address, token) as client:
            capabilities = client.capabilities()
        for run in range(2):
            tick = time.perf_counter()
            with Client(args.address, token) as client:
                images = [client.upload(path) for path in paths]
                job = client.embed(images)
                uploads = list(client.transfers)
            with Client(args.address, token) as client:
                job = client.attach(job.id).wait(timeout=180)
                job.download(directory / f"embeddings-{run}.npy")
                records.append({"job_id": job.id, "execution_device": job.data["result"]["metadata"]["execution_device"],
                                "result_shape": job.data["result"]["metadata"]["shape"], "timings": job.timings,
                                "client_end_to_end_s": time.perf_counter() - tick,
                                "uploaded_payload_bytes": sum(image.uploaded_bytes for image in images),
                                "transfers": uploads + client.transfers})
        np.testing.assert_allclose(np.load(directory / "embeddings-0.npy", allow_pickle=False),
                                   np.load(directory / "embeddings-1.npy", allow_pickle=False), rtol=1e-4, atol=1e-4)
    assert records[1]["uploaded_payload_bytes"] == 0
    assert records[1]["timings"]["model_cache_hit"] is True
    report = {"workload": "image.embed", "worker": capabilities["device"], "runs": records,
              "repeat_outputs_match": True, "note": "The first run is cold only if this executor has not loaded the encoder yet."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
