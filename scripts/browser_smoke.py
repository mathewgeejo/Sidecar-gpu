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
    parser.add_argument("--channel", default="msedge")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    token = args.token_file.read_text().strip()
    with Client(args.address, token) as client:
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
        page.locator(f'tr[data-job="{job.id}"] .job-id').click()
        assert job.id in page.locator("#job-detail").inner_text()
        with page.expect_download() as download:
            page.locator(f'tr[data-job="{job.id}"] .actions button').click()
        download.value.save_as(args.output / "browser-result.npy")
        np.testing.assert_array_equal(np.load(args.output / "browser-result.npy", allow_pickle=False), np.eye(4, dtype=np.float32))
        page.screenshot(path=str(args.output / "desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(args.output / "mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.get_by_role("button", name="Disconnect", exact=True).click()
        assert page.locator("#pair-form").is_visible()
        assert not page.locator("#console").is_visible()
        assert not errors, errors
        browser.close()
    report = {"browser": args.channel, "real_api": True, "invalid_token_rejected": True,
              "authenticated_state_matches_api": True, "result_download_verified": True,
              "mobile_has_no_page_overflow": True, "disconnect_verified": True, "page_errors": errors}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
