"""Run on computer A. All demonstration results come from computer B's API."""
import argparse
import json
import os
from pathlib import Path
from sidecar_gpu.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True, help="http://COMPUTER_B_IPV4:8377")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("remote-cuda-demo.json"))
    args = parser.parse_args()
    token = os.environ.get("SIDECAR_TOKEN") or (args.token_file.read_text().strip() if args.token_file else None)
    if not token:
        parser.error("provide --token-file or SIDECAR_TOKEN")
    report = run_benchmark(args.address, token, size=args.size, repeats=2, require_cuda=True, output=args.output)
    print(json.dumps(report, indent=2))
    print("CUDA result, reconnect, numerical correctness and zero-byte repeated inputs verified.")
    print("Confirm that the supplied address belongs to the second physical computer; an API address alone cannot prove physical separation.")


if __name__ == "__main__":
    main()
