import json
import os
from pathlib import Path
from typing import Optional
import typer
from .config import Config, load_token

app = typer.Typer(help="Sidecar GPU: durable registered workloads over a trusted LAN", no_args_is_help=True)


def emit(value):
    typer.echo(json.dumps(value, indent=2))


def client_token(token_file):
    token = os.environ.get("SIDECAR_TOKEN")
    if token:
        return token
    if token_file:
        return Path(token_file).expanduser().read_text().strip()
    raise typer.BadParameter("set SIDECAR_TOKEN or supply --token-file (never put a token in a URL)")


@app.command()
def worker(config: Optional[Path] = None, data_dir: Optional[Path] = None, device: Optional[str] = None,
           host: str = "127.0.0.1", port: int = 8377, lan: bool = False, token_file: Optional[Path] = None,
           enable_embeddings: bool = False):
    """Start the API and one spawned executor. --lan explicitly exposes port 8377."""
    import uvicorn
    from .api import create_app
    if host not in ("127.0.0.1", "localhost", "::1") and not lan:
        raise typer.BadParameter("binding beyond localhost requires --lan")
    if lan and host == "127.0.0.1":
        host = "0.0.0.0"
    settings = Config.load(config, data_dir=str(data_dir) if data_dir else None, device=device)
    if enable_embeddings:
        settings.enable_embeddings = True
    token, path = load_token(settings, token_file)
    typer.echo(f"Pairing token file: {path} (share privately; token is never logged)")
    typer.echo(f"Device requested: {settings.device}; dashboard: http://{host}:{port}")
    uvicorn.run(create_app(settings, token), host=host, port=port, access_log=False, workers=1)


@app.command()
def doctor():
    """Check this machine. This diagnostic may initialize CUDA in its own CLI process."""
    import platform
    import psutil
    info = {"hostname": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
            "host_ram_bytes": psutil.virtual_memory().total}
    try:
        import torch
        info.update(torch=torch.__version__, cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                    device_count=torch.cuda.device_count())
        info["gpus"] = [{"name": torch.cuda.get_device_name(i), "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                        for i in range(torch.cuda.device_count())]
    except ImportError:
        info["torch"] = "not installed; client-only environment"
    emit(info)


@app.command()
def prepare_model(config: Optional[Path] = None, data_dir: Optional[Path] = None):
    """Download and verify the fixed encoder weights in advance (internet required once)."""
    from .models import prepare_weights, MODEL_ID
    settings = Config.load(config, data_dir=str(data_dir) if data_dir else None)
    typer.echo("Checking/downloading encoder weights; execution jobs never download weights.")
    path = prepare_weights(settings)
    emit({"model": MODEL_ID, "state": "weights_ready", "path": str(path), "bytes": path.stat().st_size})


@app.command()
def check(address: str, token_file: Optional[Path] = None):
    """Check authenticated connectivity to an explicit worker address."""
    from .client import Client
    with Client(address, client_token(token_file)) as client:
        emit(client.capabilities())


@app.command()
def upload(address: str, path: Path, token_file: Optional[Path] = None):
    from dataclasses import asdict
    from .client import Client
    with Client(address, client_token(token_file)) as client:
        emit(asdict(client.upload(path)))


@app.command()
def jobs(address: str, job_id: Optional[str] = None, token_file: Optional[Path] = None, cancel: bool = False,
         wait: bool = False, download: Optional[Path] = None, key: Optional[str] = None):
    from .client import Client
    with Client(address, client_token(token_file)) as client:
        if not job_id and not key:
            emit(client._json("GET", "/jobs"))
            return
        job = client.find_submission(key) if key else client.attach(job_id)
        if cancel:
            job.cancel()
        if wait:
            job.wait()
        if download:
            job.download(download)
        emit(job.data)


@app.command()
def benchmark(address: str, token_file: Optional[Path] = None, size: int = 512, repeats: int = 2,
              require_cuda: bool = False, output: Path = Path("benchmark.json")):
    from .benchmark import run_benchmark
    emit(run_benchmark(address, client_token(token_file), size=size, repeats=repeats, require_cuda=require_cuda, output=output))


@app.command()
def demo(address: str, token_file: Optional[Path] = None, require_cuda: bool = False, output: Path = Path("demo.json")):
    """Upload, submit, disconnect, reattach, validate, and prove cached-input reuse."""
    from .benchmark import run_benchmark
    emit(run_benchmark(address, client_token(token_file), require_cuda=require_cuda, output=output))
