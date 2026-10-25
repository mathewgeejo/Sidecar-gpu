"""Optional real-browser checks against an already running worker (no mocked API)."""
import argparse
import json
from pathlib import Path
import numpy as np
from sidecar_gpu.client import Client


def main():
    from playwright.sync_api import sync_playwright
    parser = argparse.ArgumentParser()
    parser.add_argument("address")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(".sidecar/browser"))
    parser.add_argument("--receipt-dir", type=Path)
    parser.add_argument("--channel", default="msedge")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    token = args.token_file.read_text().strip()
    with Client(args.address, token, receipt_dir=args.receipt_dir) as client:
        a = client.upload_array(np.eye(4, dtype=np.float32))
        job = client.multiply(a, a).wait()
        count = client._json("GET", "/metrics")["object_count"]
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1100}, accept_downloads=True)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.address)
        page.locator("#token").fill("invalid-token")
        page.get_by_role("button", name="Connect").click()
        page.locator("#pair-error").get_by_text("valid bearer token required").wait_for()
        page.locator("#token").fill(token)
        page.get_by_role("button", name="Connect").click()
        page.locator("#console").wait_for(state="visible")
        assert page.locator("#object-count").inner_text() == str(count)
        assert page.locator("#mode-label").inner_text() in ("CPU execution", "CUDA execution")
        browser_runs = []
        for _ in range(2):
            page.get_by_role("button", name="Run a workload", exact=True).click()
            page.locator("#matrix-size").select_option("128")
            page.get_by_role("button", name="Run matrix job", exact=True).click()
            page.locator('#run-outcome[data-state="verified"]').wait_for(timeout=45000)
            browser_runs.append(int(page.locator("#run-uploaded").get_attribute("data-bytes")))
        assert browser_runs[1] == 0, browser_runs
        page.locator("#job-search").fill(job.id)
        assert page.locator("#job-rows tr").count() == 1
        page.locator("#job-search").fill("")
        page.locator(f'tr[data-job="{job.id}"] .job-id').click()
        assert job.id in page.locator("#job-detail").inner_text()
        with page.expect_download() as download:
            page.locator(f'tr[data-job="{job.id}"] .actions button').click()
        download.value.save_as(args.output / "browser-result.npy")
        np.testing.assert_array_equal(np.load(args.output / "browser-result.npy", allow_pickle=False), np.eye(4, dtype=np.float32))
        page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
        page.screenshot(path=str(args.output / "desktop.png"), full_page=True)
        page.screenshot(path=str(args.output / "overview.png"))
        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
        page.screenshot(path=str(args.output / "mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.emulate_media(reduced_motion="reduce")
        assert page.locator(".flow-dash").first.evaluate("el => getComputedStyle(el).animationName") == "none"
        page.get_by_role("button", name="Disconnect", exact=True).click()
        assert page.locator("#pair-form").is_visible()
        assert not page.locator("#console").is_visible()
        assert not errors, errors
        fallback = browser.new_context(viewport={"width": 1000, "height": 800})
        fallback.add_init_script("Object.defineProperty(window.crypto, 'subtle', {value: undefined});")
        fallback_page = fallback.new_page()
        fallback_page.on("pageerror", lambda error: errors.append(str(error)))
        fallback_page.goto(args.address)
        vectors = fallback_page.evaluate("""async () => {
            const {sha256} = await import('/workload.js');
            return [await sha256(new Uint8Array()), await sha256(new TextEncoder().encode('abc'))];
        }""")
        assert vectors == ["e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                           "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"]
        fallback_page.locator("#token").fill(token)
        fallback_page.get_by_role("button", name="Connect", exact=True).click()
        fallback_page.locator("#console").wait_for(state="visible")
        fallback_page.get_by_role("button", name="Run a workload", exact=True).click()
        fallback_page.locator("#matrix-size").select_option("128")
        fallback_page.get_by_role("button", name="Run matrix job", exact=True).click()
        fallback_page.locator('#run-outcome[data-state="verified"]').wait_for(timeout=45000)
        assert not errors, errors
        fallback.close()
        browser.close()
    report = {"browser": args.channel, "real_api": True, "invalid_token_rejected": True,
              "authenticated_state_matches_api": True, "result_download_verified": True,
              "mobile_has_no_page_overflow": True, "disconnect_verified": True, "page_errors": errors,
              "browser_matrix_verified": True, "repeat_browser_payload_bytes": browser_runs[1],
              "job_search_verified": True, "reduced_motion_verified": True,
              "http_lan_hash_fallback_verified": True}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
