# Acceptance evidence

Execution environment inspected on 2026-10-24: Dell Inspiron 15 3511, Windows 11, 8,320,221,184 bytes physical memory, Intel UHD Graphics, no visible NVIDIA GPU, no `nvidia-smi`, no WSL. Python 3.13.2 and the isolated environment's PyTorch 2.7.1+cpu report CUDA unavailable. The Lenovo RTX 3050 was described by the user but was not accessible for these runs. Outbound HTTPS to PyPI worked; Wi-Fi IPv4 was `10.10.19.130`. No second-machine reachability is inferred from that address.

## Automated acceptance criteria

Final run: **36 passed**, one third-party Starlette/AnyIO deprecation warning, 275.18 seconds, with offline encoder reference tests enabled. Raw JUnit evidence is `.sidecar/test-results.xml`. No frontend build was required; one editable package installation prepared the runnable CLI.

| Requested criterion | Verification | Evidence |
| --- | --- | --- |
| 1. Missing/invalid authentication | Verified | API tests reject missing/wrong bearer tokens for data/resource endpoints; browser wrong-token check |
| 2. Invalid dimensions/unsupported input types | Verified | Shape/header/payload checks; float64/int/object/pickle/Fortran/truncated/trailing/huge-shape rejection; mismatched inner dimensions and unsupported operations |
| 3. Object hash validation/deduplication | Verified | Wrong hash rejected; repeated bytes share a handle; real HTTP cached retries send zero payload bytes |
| 4. Idempotency/conflicting key reuse | Verified | Same submission returns one identity even when queue is full; changed parameters conflict; timeout SDK checks key without a second POST |
| 5. Successful result retrieval | Verified on CPU | Real worker/executor/client HTTP test, CLI benchmarks, SDK checks downloaded SHA-256, browser downloads real NPY |
| 6. Queued-job cancellation | Verified | Deterministic queue tests cancel before claim and release reservations; cancellation-request state remains nonterminal until a safe finish boundary |
| 7. Interrupted reconciliation after restart | Verified | Persisted running/cancel-requested states reconcile to interrupted; corrupt queued inputs fail; full worker restart recovers queued jobs and retains completed results |
| 8. Correct numerical output | Verified on CPU | NumPy matrix reference and separate MobileNet CPU reference, `rtol=1e-4`, `atol=1e-4`; warm encoder repeats agree |
| 9. Upload/queue limits | Verified | Declared/streamed byte limits, output limits, bounded queue, preparse JSON cap, protected-object storage pressure |

Additional verified behavior: executor-kill detection/replacement followed by a successful HTTP job; single-worker directory lock; output eviction returns 410; active input/download protection; live API uploads keep reservations during executor replacement; low-memory admission reports a failed job; CUDA worker startup refuses to fall back in this CPU-only environment; missing encoder weights fail honestly; bounded image decode/batches; local receipts survive submission response uncertainty; SDK wait timeout does not cancel work.

The restart scenario deliberately seeds a persisted running state while the executor is stopped, then starts a real worker. This verifies reconciliation without claiming a GPU kernel was killed. The executor replacement test kills an idle CPU executor. Cancellation races are checked transactionally; CUDA kernel interruption behavior is not hardware verified.

## Four distinct verification levels

1. **Unit and in-process API integration:** authentication, validation, storage, state transitions, limits, failure semantics, SDK error paths and optional image validation.
2. **Separate processes on one host:** real Uvicorn HTTP worker, spawned persistent PyTorch CPU executor, SDK/CLI client; reconnect, cache reuse, correctness, full worker restart and executor replacement. Edge headless browser tests use the real same-host API and data.
3. **Two physical computers:** **not verified**. The user deferred Lenovo setup; no reachable authenticated second worker was supplied.
4. **Actual remote CUDA:** **not verified**. No CUDA device or CUDA PyTorch build is available here. No CUDA benchmark or speedup result is reported.

## Physical demo requirements

| Required demonstration | Current evidence | Remaining verification |
| --- | --- | --- |
| A submits to B | Same-host HTTP verified | Run `scripts/two_machine_demo.py` from the actual second computer |
| B executes on CUDA | Device provenance implemented; missing-CUDA fail-fast verified | Lenovo `doctor`, CUDA result/device/events and GPU memory readings |
| Valid result returns | Matrix and embedding CPU correctness verified | Repeat numerical check against actual CUDA output |
| Second job reuses input payload | Zero-byte repeated uploads measured over loopback | Repeat across physical LAN |
| Disconnect after acceptance and reattach | Real client connection recreation and durable lookup verified; separate `detach.py` provided | Demonstrate from A to B |
| Transfer-inclusive end-to-end timing | Recorded for loopback CPU runs | Record actual LAN/CUDA run, without mixing host clocks |
| Understandable failure states | Auth, validation, admission, cancellation and interruption checks | Observe representative states during the physical demo |

## Recorded measurements and UI

- `benchmarks/cpu-loopback-512.json`: three actual matrix runs; initial input payload 2,097,408 bytes, then zero for both repeats. Exact durations, worker identity and tolerances are recorded. No speedup interpretation is applied.
- `benchmarks/cpu-embeddings.json`: two actual HTTP encoder runs; `[2,576]` outputs, approximately 2.61 s initial load, zero warm-load duration with an explicit model-cache hit; second image upload payload zero. Simple synthetic images exercise the runtime, not embedding relevance.
- `benchmarks/browser-smoke.json`: Edge checks pass for authentication, real API state, result download correctness, disconnect, no page errors and mobile page overflow. Screenshots are in `.sidecar/browser`. Browser viewport checks do not imply broad cross-browser/accessibility certification.
- Test command: `python -m pytest -q`; offline encoder HTTP/reference tests additionally use `SIDECAR_TEST_WEIGHTS` pointing to the checkpoint downloaded by `prepare-model`. Standard environments without optional dependencies/weights skip the corresponding tests explicitly.

Linux/WSL2, native Windows CUDA, firewall changes, remote TLS/tunnel setup, kernel cancellation, CUDA timings and device-memory admission are documented/implemented paths requiring the unavailable hardware or deployment environment. No firewall, driver, WSL, or system network settings were modified here.
