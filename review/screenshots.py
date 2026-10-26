"""Take phone and desktop screenshots of every page of the app -> screenshots/

  source ~/waterenv/bin/activate
  streamlit run app.py --server.headless true --server.port 8599     (in one terminal)
  python3 review/screenshots.py                                       (in another)

Needs Playwright (a tool that drives a real browser):
  pip install playwright && python3 -m playwright install chromium
If the browser won't start because of missing libraries (libnss3, libnspr4, libasound2),
run `sudo python3 -m playwright install-deps chromium` once.
"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8599"
PAGES = ["", "after-rain", "beach-profile", "cleanup-projects", "fire-impacts", "forecast", "methods"]  # "" = Home
SIZES = {"phone": (390, 844), "desktop": (1440, 900)}   # iPhone 14 width; a laptop
OUT = Path(__file__).resolve().parent.parent / "screenshots"
OUT.mkdir(exist_ok=True)


def shoot(page, name, width):
    """Wait for the page to finish drawing, then save the whole page (not just the screen)."""
    page.wait_for_selector(".finding, .home-hero", timeout=90_000)  # every page opens with a headline
    page.wait_for_timeout(5000)                           # charts and maps draw after the text
    # Streamlit scrolls inside its own box, so make the window as tall as the page instead.
    # Maps load last and can make the page taller, so repeat until the height stops changing.
    height = 0
    for _ in range(6):
        new = page.evaluate("Math.max(document.querySelector('[data-testid=\"stMain\"]')?.scrollHeight || 0,"
                            " document.body.scrollHeight)")
        if new == height:
            break
        height = new
        page.set_viewport_size({"width": width, "height": height + 40})
        page.wait_for_timeout(2500)
    page.screenshot(path=OUT / f"{name}.png", full_page=True)
    print("saved", name, height)


with sync_playwright() as p:
    browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])  # WebGL for maps
    for size, (w, h) in SIZES.items():
        ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=2 if size == "phone" else 1,
                                  is_mobile=size == "phone", has_touch=size == "phone")
        page = ctx.new_page()
        for i, path in enumerate(PAGES):
            page.set_viewport_size({"width": w, "height": h})
            page.goto(f"{URL}/{path}")
            shoot(page, f"{i}_{path or 'home'}_{size}", w)
        # The forecast's replay toggle, switched on.
        page.set_viewport_size({"width": w, "height": h})
        page.goto(f"{URL}/forecast")
        page.wait_for_selector(".finding", timeout=90_000)
        page.get_by_text("Replay the January 2023 storm").click()
        page.wait_for_selector("text=replay", timeout=60_000)
        shoot(page, f"5b_forecast-replay_{size}", w)
        ctx.close()
    browser.close()
