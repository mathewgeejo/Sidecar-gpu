"""Submit a job and exit immediately after durable acceptance."""
import argparse
import os
from pathlib import Path
import numpy as np
from sidecar_gpu.client import Client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args()
    token = os.environ.get("SIDECAR_TOKEN") or args.token_file.read_text().strip()
    with Client(args.address, token) as worker:
        rng = np.random.default_rng(100)
        a = worker.upload_array(rng.random((1024, 512), dtype=np.float32))
        b = worker.upload_array(rng.random((512, 256), dtype=np.float32))
        job = worker.multiply(a, b)
        print(job.id)
        print("Accepted. This client is exiting. Reattach using sidecar jobs --job-id.")


if __name__ == "__main__":
    main()
