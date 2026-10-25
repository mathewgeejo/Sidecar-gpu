# Sidecar GPU

A small Python runtime that executes registered matrix and image workloads on a second computer over HTTP. The worker owns one persistent execution process, keeps encoder models loaded, and stores reusable inputs and durable job identities on disk.

Start with [how Sidecar works](docs/HOW_IT_WORKS.md) for plain-language and technical explanations of the protocol, PyTorch execution, CPU versus CUDA, caching, and recovery.

**Verification so far:** native Windows, CPU execution, real HTTP, separate API/executor/client processes, and TCP network fault injection on one Dell host. The latest suite passed all 50 tests. Actual two-computer CUDA execution remains pending access to a second CUDA PC. See [the acceptance report](docs/acceptance.md). No GPU speedup is claimed.

To create a clean source archive for the CUDA PC, run `python scripts/package_source.py`. It writes `.sidecar/sidecar-gpu-source.zip`, excluding environments, runtime objects, tokens, and downloaded weights. Extract it on the other machine, then run the setup below from its `sidecar-gpu` folder.

## Quick start: one host, explicit CPU testing

Use Python **3.11–3.13**, 64-bit. The tested interpreter is Python 3.13.2 on Windows 11. Run from the repository root. The client needs no PyTorch, NVIDIA driver, or GPU.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install --no-build-isolation -e .
.\.venv\Scripts\python -m sidecar_gpu doctor
.\.venv\Scripts\python -m sidecar_gpu worker --device cpu
```

In another PowerShell terminal:

```powershell
.\.venv\Scripts\python -m sidecar_gpu demo http://127.0.0.1:8377 --token-file "$HOME\.sidecar-gpu\pairing.token" --output cpu-demo.json
```

Open **http://127.0.0.1:8377/** and paste the token from the file printed at startup. Read it privately with `Get-Content "$HOME\.sidecar-gpu\pairing.token"`. The worker never prints the token. The dashboard keeps it only in tab memory. Default worker storage is outside the repository at `~/.sidecar-gpu`.

The demo uploads two matrices, submits a job, closes the client, reconnects by ID, downloads the result, validates a NumPy CPU reference, then repeats using cached inputs. JSON includes bytes actually sent, transfer durations, execution timings, correctness, and reconnect evidence. A repeated run may find its inputs cached already; inspect the hit flags rather than assuming a cold cache.

## Simulate network faults on this Windows PC

Run independent client, worker/executor, and Toxiproxy processes over real TCP with synthetic latency, jitter, bandwidth limits, interrupted transfers, lost acceptance responses, outages, and worker restarts:

```powershell
.\.venv\Scripts\python.exe scripts/network_lab.py --download --browser
```

This downloads a pinned, checksum-verified Windows x64 Toxiproxy executable into `.sidecar/tools`. Later runs can omit `--download`. Omit `--browser` if Playwright/Edge are unavailable. Artifacts go to a fresh directory under `.sidecar/network-lab`; the script stops its own processes afterward. It makes no firewall or system-network changes. See [the network lab](docs/network-lab.md) for scenarios, evidence, and limits. Shared-host CPU tests cannot reproduce physical networking or validate CUDA.

## Two physical computers: Windows CUDA worker

On **computer B (Windows/NVIDIA worker)**, install a compatible NVIDIA driver, copy this repository, then:

```powershell
nvidia-smi
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install --no-build-isolation -e .
.\.venv\Scripts\python -m sidecar_gpu doctor
Get-NetIPAddress -AddressFamily IPv4
.\.venv\Scripts\python -m sidecar_gpu worker --device cuda --lan
```

`doctor` must report `cuda_available: true` and identify the actual GPU. The worker defaults to CUDA and exits on CUDA initialization failure; it never silently switches to CPU. The CUDA wheel includes its runtime dependencies; building CUDA kernels or installing a standalone CUDA toolkit is unnecessary for these workloads. An adequate NVIDIA driver is still required. See the [official PyTorch 2.7.1 CPU/CUDA installation commands](https://pytorch.org/get-started/previous-versions/#v271).

For the RTX 5060 and other RTX 50-series (Blackwell) GPUs, use the `cu128` wheel shown above. See [PyTorch's Blackwell support announcement](https://pytorch.org/blog/pytorch-2-7/) and [NVIDIA's compute capability table](https://developer.nvidia.com/cuda/gpus). Actual execution on the test PC still needs verification.

Use a trusted network profile. If Windows Firewall blocks the selected port, an administrator can add a scoped rule, replacing the example address with **computer A's actual IPv4**:

```powershell
New-NetFirewallRule -DisplayName "Sidecar GPU trusted client" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8377 -RemoteAddress 10.10.19.130 -Profile Private
```

The observed Dell network was Public, so that Private-only rule would not apply there. Choose the correct profile deliberately; do not disable the firewall. Remove this rule afterward with `Remove-NetFirewallRule -DisplayName "Sidecar GPU trusted client"`.

Copy the pairing token privately into a file on **computer A (client)** outside this repository. Install only the SDK there:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -e .
Test-NetConnection COMPUTER_B_IPV4 -Port 8377
.\.venv\Scripts\python -m sidecar_gpu check http://COMPUTER_B_IPV4:8377 --token-file "$HOME\sidecar-worker.token"
.\.venv\Scripts\python scripts\two_machine_demo.py --address http://COMPUTER_B_IPV4:8377 --token-file "$HOME\sidecar-worker.token" --output remote-cuda-demo.json
```

Replace `COMPUTER_B_IPV4` with its reachable Wi-Fi/Ethernet address. `127.0.0.1` refers to the current computer. The demo requires CUDA in both capabilities and result metadata. Confirm physical separation yourself: two HTTP processes cannot prove they occupy different computers. The client gets its result exclusively from the API; NumPy runs afterward to validate it.

## Linux and WSL2

Native Linux or WSL2 can use the same worker. On Ubuntu, install Python/venv and a working NVIDIA driver before creating the environment. For WSL2, install the NVIDIA **Windows** driver; do not install a Linux display driver inside WSL. Follow [NVIDIA's CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install --no-build-isolation -e .
nvidia-smi
.venv/bin/python -m sidecar_gpu doctor
.venv/bin/python -m sidecar_gpu worker --device cuda --lan
```

**WSL2 localhost forwarding does not establish reachability from computer A.** With default NAT networking, configure a Windows port proxy to the WSL IPv4 plus the Windows firewall rule. Example administrator PowerShell, replacing both example addresses with actual values:

```powershell
wsl hostname -I
netsh interface portproxy add v4tov4 listenaddress=WINDOWS_LAN_IPV4 listenport=8377 connectaddress=WSL_IPV4 connectport=8377
```

Computer A connects to `http://WINDOWS_LAN_IPV4:8377`. WSL's address can change after restart; update the proxy. Remove it using `netsh interface portproxy delete v4tov4 listenaddress=WINDOWS_LAN_IPV4 listenport=8377`. Alternatively, supported Windows 11/WSL versions offer mirrored networking with separate Hyper-V firewall considerations. Consult [Microsoft's WSL networking guide](https://learn.microsoft.com/en-us/windows/wsl/networking), and always test the port **from computer A**. Linux, WSL2, and the firewall commands are documented paths, not verified environments in this session.

## Python SDK

```python
import numpy as np
from sidecar_gpu.client import Client

with Client("http://192.168.1.20:8377", token="your-private-token") as worker:
    a = worker.upload_array(np.ones((128, 64), dtype=np.float32))
    b = worker.upload_array(np.ones((64, 32), dtype=np.float32))
    job = worker.multiply(a, b, idempotency_key="my-application-job-001")
    job_id = job.id                         # acceptance is already durable

with Client("http://192.168.1.20:8377", token="your-private-token") as worker:
    job = worker.attach(job_id).wait(timeout=120)
    print(job.timings)
    print(job.result_object())              # keep the output remote
    job.download("result.npy")              # verifies SHA-256 after download
```

`upload(path)` performs a hash-existence lookup before sending a payload. Matrices must be C-contiguous little-endian float32, two-dimensional, positive-sized NPY v1/v2 files. Pickle, NPZ, structured/object dtypes, and trailing/truncated payloads are rejected. `upload_array` saves a contiguous safe NPY to temporary disk. Raw file uploads stream through HTTPX and the API; complete upload bodies are never accumulated in RAM.

`wait()` timeout leaves the accepted job running. `find_submission(key)` recovers a lost response. The SDK fsyncs a submission receipt **before** sending, under `~/.sidecar-gpu-client/receipts`; receipts contain IDs/parameters/address, never the token. `SubmissionUncertain` includes the key and receipt location. After a 404 from lookup, retry the exact original submission with the **same key**. Changed parameters with that key conflict. `retry(job_id)` explicitly creates a new attempt for failed, interrupted, or cancelled work. SDK network calls have their own timeout; a poll already in flight may finish after the `wait` deadline.

## Optional image embeddings

The fixed encoder is torchvision MobileNet V3 Small, ImageNet1K V1 weights, with the classifier removed: **576 float32 features per image**. It uses the weight preset's resize/crop/normalization after EXIF transpose and RGB conversion. Features are not L2-normalized. Model identity is `mobilenet_v3_small.imagenet1k_v1.features576`; this is a lightweight image encoder, not a text/image semantic model.

Install matching torchvision for the worker's CPU or CUDA index, then download weights **before** the offline demonstration:

```powershell
# CUDA worker; replace cu128 with cpu for the CPU test environment.
.\.venv\Scripts\python -m pip install torchvision==0.22.1 pillow==11.3.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m sidecar_gpu prepare-model
.\.venv\Scripts\python -m sidecar_gpu worker --device cuda --lan --enable-embeddings
```

Use the same `--data-dir` for preparation and execution if overriding the default. The approximately 10 MB setup artifact lives in `data_dir/weights`, outside the managed object budget. Jobs never access the internet or accept uploaded weight files. The fixed published checkpoint checksum is verified; loading explicitly uses `weights_only=True`. Missing weights produce an honest failed job; matrix execution still works.

```python
with Client(address, token) as worker:
    images = [worker.upload("photo1.jpg"), worker.upload("photo2.png")]
    job = worker.embed(images).wait()
    job.download("embeddings.npy")  # shape: [2, 576]
```

JPEG/PNG only, one frame, at most 4096 per side, 16,777,216 decoded pixels, and 16 images per job by default. Images remain on disk between jobs; model tensors remain in the persistent executor. The model loads lazily once per executor lifetime. Result metadata identifies the model/device; timings expose first-load duration and subsequent model cache hits. Run `python examples/embeddings.py --address ... --token-file ... --images photo1.jpg photo2.png`. Omitting images generates simple test patterns and requires Pillow on that client. See [torchvision's documented model and preprocessing](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.mobilenet_v3_small.html).

## Configuration and retention

Pass `--config config.example.json`; `--device` and `--data-dir` override file settings. One process may own a data directory at a time. The SQLite database must live on a **local** filesystem; network-shared SQLite files are unsupported. Stop the worker before copying/backing up its directory.

Defaults: one executing job, eight waiting jobs, 128 MiB maximum upload, 128 MiB output payload, 8192 maximum matrix dimension, 4 GiB object/temp/output reservations. Admission checks available host memory and, for CUDA, available GPU memory with configurable headroom. Estimates are conservative, not a memory guarantee; allocation failures become visible failed jobs. No background GPU occupancy measurement is invented.

Objects are immutable and keyed by their verified SHA-256, with opaque external handles. A lookup/upload gives input objects a five-minute grace period. Queued/running/cancellation-requested jobs pin all their inputs. Successful outputs are protected for 24 hours (`retain_output=True`); `False` protects them for ten minutes to allow download. After protection expires, least-recently-accessed objects become **eligible** for eviction when new reservations need space. Expiry is not immediate deletion. Downloads lease files for up to an hour. Evicted result requests return `410 object_expired`; durable job history still identifies the successful attempt. Old handles are restored if the same bytes are uploaded again.

Upload temporary files and queued outputs reserve budget up front. Queue overflow returns `429 queue_full`; storage exhaustion returns `507 storage_full`. Protection can intentionally prevent acceptance of more work. Physical filesystem exhaustion can still fail a write. Setup weights, SQLite history/WAL, logs, and client receipts are outside the object budget and require ordinary disk maintenance. Job metadata is retained indefinitely in this prototype.

## Failure semantics

Jobs use `queued`, `running`, `succeeded`, `failed`, `cancel_requested`, `cancelled`, and `interrupted`. Acceptance, idempotency, queue admission, output reservation, and input references commit before the response. Status and result retrieval survive API restarts.

After restart, queued jobs recover only after input format/hash validation. Previously running work becomes interrupted; it is never claimed to resume. The supervisor also replaces a dead executor and reconciles its active job. Queued-job cancellation is immediate. Running cancellation is cooperative at model/input/transfer/kernel/commit boundaries; an already launched GPU kernel finishes before cancellation becomes terminal. Stopping the worker allows a short graceful interval, then may terminate its executor; unresolved work becomes interrupted. Queued jobs remain recoverable.

## CLI and tests

```powershell
python -m sidecar_gpu --help
python -m sidecar_gpu check http://WORKER_IP:8377 --token-file PATH
python -m sidecar_gpu upload http://WORKER_IP:8377 matrix.npy --token-file PATH
python -m sidecar_gpu jobs http://WORKER_IP:8377 --job-id JOB_ID --token-file PATH
python -m sidecar_gpu jobs http://WORKER_IP:8377 --key IDEMPOTENCY_KEY --token-file PATH
python -m sidecar_gpu jobs http://WORKER_IP:8377 --job-id JOB_ID --cancel --token-file PATH
python -m sidecar_gpu jobs http://WORKER_IP:8377 --job-id JOB_ID --wait --download result.npy --token-file PATH
python -m sidecar_gpu benchmark http://WORKER_IP:8377 --token-file PATH --size 1024 --repeats 3 --require-cuda --output benchmark.json
python examples/correctness.py --token-file PATH
python -m pytest -q
```

`SIDECAR_TOKEN` can replace `--token-file`; do not commit environment files containing it. Matrix comparisons use `rtol=1e-4, atol=1e-4`; GPU and CPU are not assumed bit-identical. TF32 is disabled in the CUDA executor for predictable float32 validation. Benchmark inputs are seeded and scaled; these tolerances do not guarantee identical behavior for every ill-conditioned user matrix.

Optional offline encoder tests require matching torchvision/Pillow and prepared weights:

```powershell
$env:SIDECAR_TEST_WEIGHTS = "$HOME\.sidecar-gpu\weights\mobilenet_v3_small-047dcff4.pth"
python -m pytest -q
```

Optional browser checks use an installed Edge/Chrome, with no browser download required:

```powershell
python -m pip install playwright==1.55.0
python scripts/browser_smoke.py http://127.0.0.1:8377 --token-file PATH --channel msedge
```

The dashboard is buildless: `dashboard.ts` intentionally uses browser-executable TypeScript syntax and is served as JavaScript. There is no frontend build or external CDN dependency. The workspace shows the reported CPU/CUDA mode, animated request activity, job search/filtering, measured timings, and object storage. **Run a workload** submits a real matrix job from the browser and checks the downloaded hash and every output value; deterministic inputs demonstrate cached reuse. The browser keeps its token in memory and saves only submission metadata in session storage for recovery. HTTP LAN origins use a tested SHA-256 fallback when Web Crypto is unavailable.

## Measurements and limitations

[Actual CPU matrix measurements](benchmarks/cpu-loopback-512.json) and [encoder measurements](benchmarks/cpu-embeddings.json) distinguish first observed execution, reused input bytes, model cache state, and client reconnects. The first request is not necessarily cold if the worker has already seen those inputs/models. CPU mode returns null for GPU-only transfer/event/memory values. Client end-to-end latency includes hashing, serialization, upload/lookups, submission, connection recreation, polling, result transfer, and verification of the download hash; it excludes the separately timed local numerical reference. `compute_s` includes synchronization; CUDA events separately report actual GPU elapsed computation. No clocks are subtracted across hosts. Queue timing uses the worker monotonic clock; recovery resets that duration to time since recovery and marks `queue_recovered`.

This is a single trusted-team namespace with one bearer token, no individual quotas or TLS termination built in. **HTTP authentication does not encrypt traffic.** Use a trusted LAN; optionally put the worker behind TLS with a reverse proxy or a private tunnel such as a properly configured WireGuard/Tailscale connection. Do not expose it directly to the public internet. Token file permissions follow the host account's ACLs on Windows; restrict them appropriately. Rotate the token by replacing its file while the worker is stopped. CORS is not enabled; the dashboard talks to its own origin.

No arbitrary code/commands, executable uploads, transparent CUDA interception, placement scheduler, LAN discovery, similarity search, or inference engine is implemented. Progress is truthful job state and stage timings, not a fabricated percentage. GPU utilization is unavailable; memory is sampled between jobs. Kernel hangs/time limits require manual intervention; execution is bounded by workload sizes, admission, queue length, and single concurrency, not a preemptive per-kernel deadline. CPU tests and browser checks cannot establish remote CUDA correctness.

See [architecture and protocol](docs/architecture.md), [acceptance evidence](docs/acceptance.md), and [the two-minute demo](docs/demo.md).
