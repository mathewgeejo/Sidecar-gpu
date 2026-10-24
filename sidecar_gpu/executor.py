"""The spawned child owns PyTorch, CUDA and any persistent workload models."""
import os
import time
from .arrays import inspect_array
from .config import Config
from .errors import SidecarError
from .store import Store


def execute_job(store, job, device, torch, models=None):
    import numpy as np
    import psutil
    config = store.config
    started = time.perf_counter()
    timings = {"model_load_s": None}
    params = job["parameters"]
    embedding = params["operation"] == "image.embed"
    models = models if models is not None else {}
    temporary = store.temp / (job["id"] + ".npy")
    a = b = ta = tb = result = output = None
    try:
        handles = params["images"] if embedding else [params["a"], params["b"]]
        objects = [store.object(handle) for handle in handles]
        if embedding:
            sizes = [len(objects) * 3 * 224 * 224 * 4]
            output_size = len(objects) * 576 * 4
            decoded = max(obj["metadata"]["width"] * obj["metadata"]["height"] * 8 for obj in objects)
            estimate = decoded + 256 * 1024**2 + 2 * (sum(sizes) + output_size)
            gpu_estimate = sum(sizes) + output_size + (64 + 16 * len(objects)) * 1024**2
        else:
            sizes = [obj["metadata"]["payload_bytes"] for obj in objects]
            output_size = params["a_shape"][0] * params["b_shape"][1] * 4
            estimate = 2 * (sum(sizes) + output_size) + 64 * 1024**2
            gpu_estimate = sum(sizes) + output_size + 64 * 1024**2
        if estimate > psutil.virtual_memory().available * config.host_memory_fraction:
            raise SidecarError("host_memory_limit", "job exceeds available host-memory admission budget", 503)
        if device == "cuda":
            free, total = torch.cuda.mem_get_info()
            if gpu_estimate > free * config.gpu_memory_fraction:
                raise SidecarError("gpu_memory_limit", "job exceeds available GPU-memory admission budget", 503)
            torch.cuda.reset_peak_memory_stats()
        if store.job(job["id"])["state"] == "cancel_requested":
            store.finish(job["id"], "cancelled", timings)
            return
        if embedding:
            from .models import load_encoder, preprocess_images, MODEL_ID
            tick = time.perf_counter()
            hit = MODEL_ID in models
            if not hit:
                store.set_runtime({**store.runtime(), "model_state": "loading", "loading_model": MODEL_ID, "sampled_at": time.time()})
                try:
                    models[MODEL_ID] = load_encoder(config, device, torch)
                except BaseException:
                    store.set_runtime({**store.runtime(), "model_state": "load_failed", "loading_model": None, "sampled_at": time.time()})
                    raise
            timings["model_load_s"] = 0.0 if hit else time.perf_counter() - tick
            timings["model_cache_hit"] = hit
            store.set_runtime({**store.runtime(), "model_state": "loaded", "model_cache": list(models), "loading_model": None, "sampled_at": time.time()})
            model, transform = models[MODEL_ID]
        tick = time.perf_counter()
        # copy=False is unsafe for read-only mmap with PyTorch; copy inputs explicitly.
        if embedding:
            a = preprocess_images(store, objects, transform, torch)
        else:
            a = np.load(store.path(objects[0]["sha256"]), allow_pickle=False)
            b = np.load(store.path(objects[1]["sha256"]), allow_pickle=False)
        timings["input_load_s"] = time.perf_counter() - tick
        tick = time.perf_counter()
        if embedding:
            ta = a.to(device)
        else:
            ta, tb = torch.from_numpy(a).to(device), torch.from_numpy(b).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        timings["host_to_device_s"] = time.perf_counter() - tick if device == "cuda" else None
        if store.job(job["id"])["state"] == "cancel_requested":
            store.finish(job["id"], "cancelled", timings)
            return
        tick = time.perf_counter()
        if device == "cuda":
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        with torch.inference_mode():
            result = model(ta) if embedding else torch.matmul(ta, tb)
        if device == "cuda":
            end.record()
            end.synchronize()
            timings["gpu_compute_s"] = begin.elapsed_time(end) / 1000
        else:
            timings["gpu_compute_s"] = None
        timings["compute_s"] = time.perf_counter() - tick
        # A launched kernel has finished before cancellation can become terminal.
        if store.job(job["id"])["state"] == "cancel_requested":
            timings["worker_total_s"] = time.perf_counter() - started
            store.finish(job["id"], "cancelled", timings)
            return
        tick = time.perf_counter()
        output = result.cpu().numpy()
        if device == "cuda":
            torch.cuda.synchronize()
        timings["device_to_host_s"] = time.perf_counter() - tick if device == "cuda" else None
        tick = time.perf_counter()
        with temporary.open("wb") as stream:
            np.save(stream, output, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        timings["serialization_s"] = time.perf_counter() - tick
        metadata = inspect_array(temporary, config, output=True)
        metadata["execution_device"] = device
        metadata["device_name"] = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
        if embedding:
            metadata.update(model=MODEL_ID, embedding_dimensions=576)
        timings["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if device == "cuda" else None
        timings["worker_total_s"] = time.perf_counter() - started
        tick = time.perf_counter()
        store.finish(job["id"], "succeeded", timings, temporary=temporary, metadata=metadata, worker_started=started)
    except Exception as exc:
        if device == "cuda":
            # Do not acknowledge cancellation while previously launched work is active.
            try:
                torch.cuda.synchronize()
            except Exception:
                raise  # Supervisor must interrupt this job and replace the damaged context.
        error = {"code": exc.code if isinstance(exc, SidecarError) else "execution_failed",
                 "message": exc.message if isinstance(exc, SidecarError) else f"{type(exc).__name__}: {str(exc)[:500]}"}
        timings["worker_total_s"] = time.perf_counter() - started
        store.finish(job["id"], "failed", timings, temporary=temporary, error=error)
    finally:
        temporary.unlink(missing_ok=True)
        del a, b, ta, tb, result, output


def executor_main(config_dict, stop, ready):
    try:
        import torch
        import psutil
        config = Config(**config_dict)
        store = Store(config)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        if config.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable. Install a CUDA PyTorch wheel and NVIDIA driver, or explicitly use --device cpu for testing.")
            torch.cuda.init()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        info = {"device": config.device, "device_name": torch.cuda.get_device_name(0) if config.device == "cuda" else "CPU",
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "executor_pid": os.getpid(), "model_cache": [], "gpu_utilization_percent": None,
                "host_memory_total_bytes": psutil.virtual_memory().total}
        ready.send({"ok": True, "info": info})
        ready.close()
        last_telemetry = 0
        # Model instances survive job boundaries, but not executor replacement.
        models = {}
        while not stop.is_set():
            if time.monotonic() - last_telemetry >= 1:
                telemetry = {**info, "sampled_at": time.time(), "model_cache": list(models),
                             "model_state": "loaded" if models else "not_loaded", "loading_model": None}
                if config.device == "cuda":
                    free, total = torch.cuda.mem_get_info()
                    telemetry.update(gpu_free_bytes=free, gpu_total_bytes=total,
                                     gpu_allocated_bytes=torch.cuda.memory_allocated(), gpu_reserved_bytes=torch.cuda.memory_reserved())
                else:
                    telemetry.update(gpu_free_bytes=None, gpu_total_bytes=None, gpu_allocated_bytes=None, gpu_reserved_bytes=None)
                store.set_runtime(telemetry)
                last_telemetry = time.monotonic()
            job = store.claim()
            if job:
                execute_job(store, job, config.device, torch, models)
            else:
                stop.wait(0.05)
    except BaseException as exc:
        try:
            ready.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        except (OSError, EOFError):
            pass
        raise
