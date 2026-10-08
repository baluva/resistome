"""Captures du dashboard (Playwright) -> docs/screenshots/. Lancer l'app sur :8531 avant."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8531"
PAGES = [("", "01_probleme"), ("outil", "02_outil"), ("carte", "03_carte"), ("validation", "04_validation"),
         ("genomes", "05_genomes"), ("surveillance", "06_surveillance"), ("modele", "07_modele"),
         ("prediction", "08_prediction"), ("methode", "09_methode")]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        for path, name in PAGES:
            pg.goto(f"{BASE}/{path}", wait_until="networkidle")
            pg.wait_for_selector('[data-testid="stAppViewContainer"]')
            pg.wait_for_timeout(1500)
            # attend que Streamlit ait fini de calculer
            for _ in range(60):
                if not pg.query_selector('[data-testid="stStatusWidget"]'):
                    break
                pg.wait_for_timeout(500)
            pg.wait_for_timeout(2500)
            pg.screenshot(path=str(OUT / f"{name}.png"), full_page=False)
            # page entière : on agrandit la zone de défilement de Streamlit
            h = pg.evaluate("document.querySelector('[data-testid=\"stMain\"]').scrollHeight")
            pg.set_viewport_size({"width": 1440, "height": min(int(h) + 40, 6000)})
            pg.wait_for_timeout(1500)
            pg.screenshot(path=str(OUT / f"{name}_full.png"), full_page=True)
            pg.set_viewport_size({"width": 1440, "height": 900})
            print("ok", name, h)
        b.close()


if __name__ == "__main__":
    main()
