"""Real TCP fault tests. Install the optional Toxiproxy binary using scripts/network_lab.py."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest
from scripts.network_lab import DEFAULT_BINARY, NetworkLab, PROFILES, ROOT


@pytest.fixture(scope="module")
def network_lab(tmp_path_factory):
    binary = Path(os.environ.get("SIDECAR_TOXIPROXY", DEFAULT_BINARY))
    if not binary.is_file():
        pytest.skip("optional network lab: run python scripts/network_lab.py --download")
    directory = os.environ.get("SIDECAR_NETWORK_LAB_DIR") or tmp_path_factory.mktemp("network-lab")
    lab = NetworkLab(directory, binary)
    try:
        lab.start()
        yield lab
    finally:
        lab.close()


@pytest.fixture
def lab(network_lab):
    network_lab.configure()
    yield network_lab
    network_lab.configure()


def key(label):
    return label + "-" + uuid.uuid4().hex


def parameters(objects):
    return {"operation": "matrix.multiply", "a": objects[0]["handle"], "b": objects[1]["handle"],
            "a_shape": objects[0]["metadata"]["shape"], "b_shape": objects[1]["metadata"]["shape"]}


@pytest.mark.integration
def test_network_authentication(lab):
    lab.record("authentication", lab.client("auth"))


@pytest.mark.integration
def test_network_latency_is_measured(lab):
    baseline = lab.client("health")
    lab.configure({"latency_ms": 50, "jitter_ms": 0})
    delayed = lab.client("health")
    assert delayed["median_s"] >= 0.08  # 50 ms each direction; allow scheduler/clock resolution.
    lab.record("latency_calibration", {"baseline": baseline, "delayed": delayed,
                                       "configured_one_way_ms": 50})


@pytest.mark.integration
@pytest.mark.parametrize("profile", PROFILES)
def test_network_profile_correctness_cache_and_process_reconnect(lab, profile):
    settings = PROFILES[profile]
    lab.configure(settings)
    inputs = lab.inputs("profile-" + profile, seed=100 + list(PROFILES).index(profile))
    runs = []
    for attempt in range(2):
        submitted = lab.client("submit", **inputs, key=key(profile))
        finished = lab.client("finish", **inputs, job_id=submitted["job_id"])
        assert submitted["pid"] != finished["pid"]
        uploaded = sum(obj["uploaded_bytes"] for obj in submitted["objects"])
        assert uploaded > 0 if attempt == 0 else uploaded == 0
        runs.append({"submitted": submitted, "finished": finished, "uploaded_payload_bytes": uploaded})
    if settings["up_kb_s"]:
        for transfer in runs[0]["submitted"]["transfers"]:
            # More than one MiB is sent; tolerate a 64 KiB initial burst and timer granularity.
            minimum = max(0, transfer["bytes"] - 65536) / (settings["up_kb_s"] * 1000)
            assert transfer["duration_s"] >= minimum * 0.8
    lab.record("profile_" + profile, {"settings": settings, "runs": runs})


@pytest.mark.integration
def test_network_outage_and_wait_timeout_do_not_cancel_accepted_work(lab):
    inputs = lab.inputs("outage", seed=200)
    try:
        with lab.paused_executor():
            submitted = lab.client("submit", **inputs, key=key("outage"))
            assert lab.observe("jobs/" + submitted["job_id"])["state"] == "queued"
            timeout = lab.client("wait_timeout", job_id=submitted["job_id"])
            lab.enabled(False)
            unavailable = lab.client("unreachable", timeout=0.5)
        # Administrative observer only: the worker completes while the client link is still down.
        completed = lab.wait_observed("jobs/" + submitted["job_id"], lambda job: job["state"] == "succeeded")
    finally:
        lab.enabled(True)
    finished = lab.client("finish", **inputs, job_id=submitted["job_id"])
    lab.record("outage_after_acceptance", {"submitted": submitted, "wait_timeout": timeout,
                                           "unavailable": unavailable, "completed_while_disconnected": completed["state"],
                                           "finished": finished, "arrangement": "idle executor paused until link disabled"})


@pytest.mark.integration
def test_network_lost_submission_response_recovers_exactly_one_identity(lab):
    inputs = lab.inputs("uncertain", seed=300)
    prepared = lab.client("prepare", **inputs)
    payload = parameters(prepared["objects"])
    identity = key("lost-response")
    before = len(lab.observe("jobs?limit=200")["jobs"])
    lab.toxic("timeout", "downstream", {"timeout": 0})
    uncertain = lab.client("uncertain", parameters=payload, key=identity, timeout=0.4)
    accepted = lab.observe("jobs/by-key/" + identity)
    lab.configure()
    recovered = lab.client("recover", parameters=payload, key=identity)
    assert recovered["job_id"] == accepted["id"]
    assert len(lab.observe("jobs?limit=200")["jobs"]) == before + 1
    finished = lab.client("finish", **inputs, job_id=recovered["job_id"])
    lab.record("lost_submission_response", {"uncertain": uncertain, "recovered": recovered,
                                             "accepted_job_id": accepted["id"], "new_job_count": 1,
                                             "finished": finished})


@pytest.mark.integration
def test_network_interrupted_upload_cleans_up_and_retries(lab):
    inputs = lab.inputs("partial-upload", seed=400)
    lab.toxic("limit_data", "upstream", {"bytes": 8192})
    broken = lab.client("broken_upload", **inputs, timeout=3)
    lab.configure()
    metrics = lab.wait_observed("metrics", lambda value: value["reserved_bytes"] == 0)
    assert not list((lab.worker_dir / "tmp").iterdir())
    sha256 = hashlib.sha256(Path(inputs["a_path"]).read_bytes()).hexdigest()
    assert lab.observe("objects/lookup/" + sha256)["exists"] is False
    retry = lab.client("upload", **inputs)
    assert retry["objects"][0]["uploaded_bytes"] == Path(inputs["a_path"]).stat().st_size
    lab.record("interrupted_upload", {"broken": broken, "retry": retry,
                                       "reserved_bytes_after_disconnect": metrics["reserved_bytes"],
                                       "partial_worker_files_removed": True})


@pytest.mark.integration
def test_network_interrupted_download_is_atomic_and_retries(lab):
    inputs = lab.inputs("partial-download", seed=500)
    submitted = lab.client("submit", **inputs, key=key("download"))
    lab.wait_observed("jobs/" + submitted["job_id"], lambda job: job["state"] == "succeeded")
    lab.toxic("limit_data", "downstream", {"bytes": 8192})
    broken = lab.client("broken_download", job_id=submitted["job_id"], timeout=3)
    lab.configure()
    finished = lab.client("finish", **inputs, job_id=submitted["job_id"])
    lab.record("interrupted_download", {"broken": broken, "finished": finished})


@pytest.mark.integration
def test_network_blackhole_has_a_bounded_read_timeout(lab):
    lab.toxic("timeout", "downstream", {"timeout": 0})
    result = lab.client("read_timeout", timeout=0.4)
    assert 0.3 <= result["elapsed_s"] < 3
    lab.record("read_timeout", result)


@pytest.mark.integration
def test_network_concurrent_duplicate_submissions(lab):
    inputs = lab.inputs("concurrent", seed=600)
    identity = key("concurrent")
    before = len(lab.observe("jobs?limit=200")["jobs"])
    pending = [lab.begin_client("submit", **inputs, key=identity) for _ in range(2)]
    results = [lab.end_client(process) for process in pending]
    assert len({result["pid"] for result in results}) == 2
    assert results[0]["job_id"] == results[1]["job_id"]
    assert len(lab.observe("jobs?limit=200")["jobs"]) == before + 1
    finished = lab.client("finish", **inputs, job_id=results[0]["job_id"])
    lab.record("concurrent_idempotency", {"clients": results, "new_job_count": 1, "finished": finished})


@pytest.mark.integration
def test_network_worker_crash_recovers_queue_and_previous_results(lab):
    from scripts.network_lab import stop_tree
    inputs = lab.inputs("restart", seed=700)
    completed = lab.client("submit", **inputs, key=key("before-crash"))
    lab.client("finish", **inputs, job_id=completed["job_id"])
    with lab.paused_executor():
        queued = lab.client("submit", **inputs, key=key("queued-at-crash"))
        assert lab.observe("jobs/" + queued["job_id"])["state"] == "queued"
        stop_tree(lab.worker)
    unavailable = lab.client("unreachable", timeout=0.5)
    lab.start_worker()
    recovered = lab.client("finish", **inputs, job_id=queued["job_id"])
    assert recovered["queue_recovered"] is True
    previous = lab.client("finish", **inputs, job_id=completed["job_id"])
    lab.record("worker_restart", {"unavailable": unavailable, "recovered": recovered,
                                   "previous_completed_result": previous,
                                   "arrangement": "idle executor paused to leave an actual accepted job queued"})


@pytest.mark.integration
def test_network_dashboard_through_shaped_link(lab):
    if os.environ.get("SIDECAR_NETWORK_LAB_BROWSER") != "1":
        pytest.skip("add --browser to scripts/network_lab.py for the optional installed-Edge check")
    pytest.importorskip("playwright")
    lab.configure(PROFILES["wifi"])
    directory = lab.client_dir / "browser"
    process = lab.launch([sys.executable, str(ROOT / "scripts" / "browser_smoke.py"), lab.address,
                          "--token-file", str(lab.client_token), "--output", str(directory),
                          "--receipt-dir", str(lab.client_dir / "receipts"), "--channel", "msedge"], "browser")
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        from scripts.network_lab import stop_tree
        stop_tree(process)
        pytest.fail("browser exceeded 60 seconds")
    assert process.returncode == 0, f"inspect {lab.logs / 'browser.log'}"
    lab.record("dashboard_shaped_link", json.loads((directory / "report.json").read_text()))
