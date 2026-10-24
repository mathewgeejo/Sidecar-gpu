# Two-minute demonstration

Prepare both environments, a reachable CUDA worker, token sharing, and a permitted firewall rule in advance. For embeddings, install matching torchvision/Pillow and run `prepare-model` before going offline. Keep the worker terminal and dashboard visible on computer B. Open the dashboard on computer A using B's actual LAN address, pair privately, and verify the displayed CUDA model.

**0:00–0:20 — Establish the actual machines.** Show computer B's `nvidia-smi` and `sidecar doctor`; show computer A's explicit worker IP and `sidecar check`. Explain that CUDA belongs to B and that the current run must report CUDA rather than CPU.

**0:20–1:05 — Execute and validate.** On A run:

```powershell
python scripts/two_machine_demo.py --address http://COMPUTER_B_IPV4:8377 --token-file PATH --output remote-cuda-demo.json
```

The client uploads matrices, receives a persisted job ID, closes its HTTP client, reattaches with a new client, downloads B's output, and checks a local NumPy reference. Show the job's real device/timings and its appearance in B's dashboard.

**1:05–1:25 — Prove reuse and reconnect.** Show the report's second job: `uploaded_payload_bytes: 0`, `disconnected_and_reattached: true`, and a new job ID. Reopen it independently with `python -m sidecar_gpu jobs ADDRESS --job-id JOB_ID --token-file PATH`. Point out that closing the original client did not cancel it.

**1:25–1:45 — Explain overhead and failures.** Compare `client_end_to_end_s`, upload/download durations, queue wait, transfer, and compute. Do not infer a speedup from a tiny matrix. Show the automated missing-token rejection and an actual failed/interrupted job from an earlier acceptance run, if available; otherwise demonstrate a wrong pairing token. No invented failure rows or utilization numbers.

**1:45–2:00 — Persistent model option.** If prepared, show the embedding report with one encoder load, zero warm-load time, cached image handles, and a `[batch,576]` result. Otherwise close with matrix correctness, durable identity, and zero-byte repeated input evidence.

The automated script closes/recreates client connections in one invocation. For an on-screen stop/restart of the entire client process, use `examples/detach.py` below, retain its printed ID, and run `sidecar jobs --job-id ... --wait --download ...` in a new terminal. Both commands use B's API; neither executes the requested operation locally.
