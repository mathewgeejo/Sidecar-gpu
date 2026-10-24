"""A small CPU reference verifies a downloaded remote result; it never supplies the result."""
import argparse
import os
from pathlib import Path
import numpy as np
from sidecar_gpu.client import Client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="http://127.0.0.1:8377")
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args()
    token = os.environ.get("SIDECAR_TOKEN") or args.token_file.read_text().strip()
    rng = np.random.default_rng(7)
    a, b = rng.random((64, 32), dtype=np.float32), rng.random((32, 16), dtype=np.float32)
    with Client(args.address, token) as worker:
        remote_a, remote_b = worker.upload_array(a), worker.upload_array(b)
        job = worker.multiply(remote_a, remote_b)
        print("Accepted:", job.id)
    with Client(args.address, token) as worker:
        job = worker.attach(job.id).wait()
        job.download("correctness-result.npy")
        actual = np.load("correctness-result.npy", allow_pickle=False)
        np.testing.assert_allclose(actual, a @ b, rtol=1e-4, atol=1e-4)
        print("Correct result from", job.data["result"]["metadata"]["execution_device"], job.timings)


if __name__ == "__main__":
    main()
