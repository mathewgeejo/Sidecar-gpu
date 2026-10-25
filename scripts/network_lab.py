"""Run real Sidecar HTTP clients through a local Toxiproxy fault injector."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
import xml.etree.ElementTree as ET

import httpx
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
VERSION = "2.13.0"
ASSET = "toxiproxy-server-windows-amd64.exe"
DOWNLOAD_URL = f"https://github.com/Shopify/toxiproxy/releases/download/v{VERSION}/{ASSET}"
# GitHub's SHA-256 release asset digest, checked against the official v2.13.0 release.
ASSET_SHA256 = "04b710eefd7690c89f7cc67289e3e14618f1ff3c26e4a5f752a34a879b895bab"
DEFAULT_BINARY = ROOT / ".sidecar" / "tools" / f"toxiproxy-{VERSION}" / ASSET
PROFILES = {
    "clean": {"latency_ms": 0, "jitter_ms": 0, "up_kb_s": 0, "down_kb_s": 0},
    "wifi": {"latency_ms": 15, "jitter_ms": 5, "up_kb_s": 2048, "down_kb_s": 6144},
    "slow": {"latency_ms": 60, "jitter_ms": 15, "up_kb_s": 256, "down_kb_s": 512},
    "fragmented": {"latency_ms": 5, "jitter_ms": 0, "up_kb_s": 0,
                   "down_kb_s": 0, "slice_bytes": 1024},
}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install_binary(path):
    if platform.system() != "Windows" or platform.machine().lower() not in ("amd64", "x86_64"):
        raise RuntimeError("automatic download supports Windows x64; supply --toxiproxy on other platforms")
    path = Path(path)
    if path.is_file() and digest(path) == ASSET_SHA256:
        print(f"Verified Toxiproxy {VERSION}: {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    try:
        request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "sidecar-network-lab"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if digest(temporary) != ASSET_SHA256:
            raise RuntimeError("Toxiproxy release checksum mismatch; downloaded file was not installed")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded and verified Toxiproxy {VERSION}: {path}", flush=True)


def free_ports(count):
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def stop_tree(process):
    """Only stop the subprocess tree created and retained by this lab instance."""
    if process is None or process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
        owned = parent.children(recursive=True) + [parent]
        for child in reversed(owned):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(owned, timeout=5)
        process.wait(timeout=10)
    except psutil.NoSuchProcess:
        process.wait(timeout=10)


class NetworkLab:
    def __init__(self, directory, binary):
        self.root = Path(directory).resolve()
        self.binary = Path(binary).resolve()
        self.client_dir = self.root / "client"
        self.worker_dir = self.root / "worker"
        self.logs = self.root / "logs"
        for directory in (self.client_dir, self.worker_dir, self.logs):
            directory.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.client_token = self.client_dir / "pairing.token"
        for path in (self.client_token, self.worker_dir / "pairing.token"):
            with path.open("x", encoding="utf-8") as stream:
                stream.write(self.token + "\n")
        self.worker_port, self.proxy_port, self.control_port = free_ports(3)
        self.worker_address = f"http://127.0.0.1:{self.worker_port}"
        self.address = f"http://127.0.0.1:{self.proxy_port}"
        self.control_address = f"http://127.0.0.1:{self.control_port}"
        self.control = httpx.Client(base_url=self.control_address, timeout=5, trust_env=False)
        self.observer = httpx.Client(base_url=self.worker_address + "/api/v1/", timeout=5,
                                     trust_env=False, headers={"Authorization": "Bearer " + self.token})
        self.worker = self.proxy = None
        self.children = []
        self.log_streams = []
        self.sequence = 0
        self.worker_generation = 0
        self.evidence = {
            "verification_level": "single host; separate processes; real TCP with injected faults",
            "physical_hosts": 1, "execution_device": "cpu", "cuda_verified": False,
            "platform": platform.platform(), "python": platform.python_version(),
            "orchestrator_pid": os.getpid(), "worker_address": self.worker_address,
            "client_address": self.address, "toxiproxy_version": VERSION,
            "toxiproxy_binary_sha256": digest(self.binary), "profiles": PROFILES,
            "client_data_dir": str(self.client_dir), "worker_data_dir": str(self.worker_dir),
            "limitations": [
                "shared host, kernel, CPU, RAM, filesystem permissions and clock",
                "TCP proxy does not emulate radio, Ethernet, IP packet loss/retransmission, routing or firewall",
                "latency affects data streams, not TCP SYN/ACK handshakes; bandwidth is per connection",
                "configured profiles are synthetic, not measurements of a particular physical network",
                "CPU worker provides no NVIDIA, CUDA performance, driver or VRAM validation",
            ],
            "scenarios": [], "worker_generations": [],
        }

    def launch(self, command, name, cwd=None, env=None):
        stream = (self.logs / f"{name}.log").open("w", encoding="utf-8")
        self.log_streams.append(stream)
        process = subprocess.Popen(command, cwd=cwd or ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   env={**os.environ, **(env or {})},
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self.children.append(process)
        return process

    def wait_ready(self, address, process, health=False):
        deadline = time.monotonic() + 90
        with httpx.Client(timeout=1, trust_env=False) as http:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"process {process.pid} exited; inspect {self.logs}")
                try:
                    response = http.get(address)
                    if response.status_code == 200 and (not health or response.json()["status"] == "ready"):
                        return
                except httpx.TransportError:
                    pass
                time.sleep(0.1)
        raise TimeoutError(f"startup timed out: {address}; inspect {self.logs}")

    def start_worker(self):
        self.worker_generation += 1
        self.worker = self.launch(
            [sys.executable, "-m", "sidecar_gpu", "worker", "--device", "cpu",
             "--data-dir", str(self.worker_dir), "--port", str(self.worker_port)],
            f"worker-{self.worker_generation}")
        self.wait_ready(self.worker_address + "/api/v1/health", self.worker, health=True)
        info = self.observe("capabilities")
        assert info["ready"] and info["device"]["device"] == "cpu"
        self.evidence["worker_generations"].append({"api_launcher_pid": self.worker.pid,
                                                   "device": info["device"]})
        self.save()

    def start(self):
        self.start_worker()
        self.proxy = self.launch([str(self.binary), "-host", "127.0.0.1", "-port", str(self.control_port)],
                                 "toxiproxy", env={"LOG_LEVEL": "info"})
        self.wait_ready(self.control_address + "/version", self.proxy)
        version = self.control.get("/version")
        version.raise_for_status()
        assert version.json()["version"].lstrip("v") == VERSION, version.text
        self.api("POST", "/proxies", {"name": "sidecar", "listen": f"127.0.0.1:{self.proxy_port}",
                                      "upstream": f"127.0.0.1:{self.worker_port}", "enabled": True})
        self.evidence["proxy_pid"] = self.proxy.pid
        self.save()

    def close(self):
        for process in reversed(self.children):
            stop_tree(process)
        self.control.close()
        self.observer.close()
        for stream in self.log_streams:
            stream.close()
        self.evidence["owned_processes_stopped"] = all(p.poll() is not None for p in self.children)
        self.save()

    def save(self):
        write_json(self.root / "evidence.json", self.evidence)

    def record(self, name, data):
        self.evidence["scenarios"].append({"name": name, **data})
        self.save()

    def api(self, method, path, payload=None):
        response = self.control.request(method, path, json=payload)
        response.raise_for_status()
        return response.json() if response.content else None

    def observe(self, path):
        response = self.observer.get(path)
        response.raise_for_status()
        return response.json()

    def enabled(self, value):
        self.api("POST", "/proxies/sidecar", {"enabled": value})

    def toxic(self, kind, stream, attributes):
        return self.api("POST", "/proxies/sidecar/toxics",
                        {"name": f"{kind}_{stream}", "type": kind, "stream": stream,
                         "toxicity": 1.0, "attributes": attributes})

    def configure(self, profile=None):
        self.enabled(False)
        for toxic in self.api("GET", "/proxies/sidecar/toxics"):
            self.api("DELETE", "/proxies/sidecar/toxics/" + toxic["name"])
        self.enabled(True)
        profile = profile or PROFILES["clean"]
        for stream, rate in (("upstream", "up_kb_s"), ("downstream", "down_kb_s")):
            if profile.get("slice_bytes"):
                self.toxic("slicer", stream, {"average_size": profile["slice_bytes"],
                                             "size_variation": profile["slice_bytes"] // 2, "delay": 0})
            if profile.get("latency_ms"):
                self.toxic("latency", stream, {"latency": profile["latency_ms"],
                                              "jitter": profile.get("jitter_ms", 0)})
            if profile.get(rate):
                self.toxic("bandwidth", stream, {"rate": profile[rate]})

    def inputs(self, name, *, seed=7, size=512):
        directory = self.client_dir / name
        directory.mkdir()
        rng = np.random.default_rng(seed)
        for label in ("a", "b"):
            array = (rng.random((size, size), dtype=np.float32) - 0.5) / np.sqrt(size)
            np.save(directory / f"{label}.npy", array.astype(np.float32), allow_pickle=False)
        return {"a_path": str(directory / "a.npy"), "b_path": str(directory / "b.npy")}

    def begin_client(self, mode, **payload):
        self.sequence += 1
        name = f"{self.sequence:03d}-{mode}"
        request_path = self.client_dir / f"{name}-request.json"
        result_path = self.client_dir / f"{name}-result.json"
        request = {"mode": mode, "address": self.address, "token_file": str(self.client_token),
                   "receipt_dir": str(self.client_dir / "receipts"),
                   "destination": str(self.client_dir / f"{name}.npy"), **payload}
        write_json(request_path, request)
        process = self.launch([sys.executable, str(ROOT / "scripts" / "network_client.py"),
                               str(request_path), str(result_path)], name, cwd=self.client_dir)
        return process, result_path

    def end_client(self, pending):
        process, result_path = pending
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            stop_tree(process)
            raise TimeoutError(f"client exceeded 60 seconds; inspect {self.logs}") from None
        if process.returncode:
            raise RuntimeError(f"client {process.pid} failed ({process.returncode}); inspect {self.logs}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["torch_imported"] is False
        assert result["pid"] != self.evidence["worker_generations"][-1]["device"]["executor_pid"]
        return result

    def client(self, mode, **payload):
        return self.end_client(self.begin_client(mode, **payload))

    @contextmanager
    def paused_executor(self):
        pid = self.observe("capabilities")["device"]["executor_pid"]
        owned = {child.pid for child in psutil.Process(self.worker.pid).children(recursive=True)}
        assert pid in owned, "refusing to suspend an executor outside this lab's process tree"
        process = psutil.Process(pid)
        process.suspend()
        try:
            yield pid
        finally:
            try:
                process.resume()
            except psutil.NoSuchProcess:
                pass

    def wait_observed(self, path, predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.observe(path)
            if predicate(value):
                return value
            time.sleep(0.05)
        raise TimeoutError(f"observer condition not met: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="download the pinned Windows x64 proxy if needed")
    parser.add_argument("--install-only", action="store_true")
    parser.add_argument("--toxiproxy", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path, default=ROOT / ".sidecar" / "network-lab")
    parser.add_argument("--browser", action="store_true", help="also test the dashboard through the shaped link in Edge")
    args = parser.parse_args()
    if args.download:
        install_binary(args.toxiproxy)
    if not args.toxiproxy.is_file():
        parser.error("Toxiproxy is missing; run again with --download")
    if args.toxiproxy.resolve() == DEFAULT_BINARY.resolve() and digest(args.toxiproxy) != ASSET_SHA256:
        parser.error("cached Toxiproxy checksum mismatch; run again with --download")
    if args.install_only:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = args.output.resolve() / stamp
    directory.mkdir(parents=True)
    environment = {**os.environ, "SIDECAR_TOXIPROXY": str(args.toxiproxy.resolve()),
                   "SIDECAR_NETWORK_LAB_DIR": str(directory),
                   "SIDECAR_NETWORK_LAB_BROWSER": "1" if args.browser else "0"}
    print(f"Network lab artifacts: {directory}", flush=True)
    command = [sys.executable, "-m", "pytest", "tests/test_network_lab.py", "-v", "-x", "--tb=short",
               "-p", "no:cacheprovider", f"--junitxml={directory / 'tests.xml'}"]
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=sys.stdout, stderr=sys.stderr,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        code = process.wait()
    except KeyboardInterrupt:
        stop_tree(process)
        raise
    report = {"exit_code": code, "artifacts": str(directory), "physical_hosts": 1, "cuda_verified": False}
    if (directory / "tests.xml").is_file():
        suite = ET.parse(directory / "tests.xml").getroot().find("testsuite")
        report.update({key: suite.attrib[key] for key in ("tests", "failures", "errors", "skipped", "time")})
    write_json(directory / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
