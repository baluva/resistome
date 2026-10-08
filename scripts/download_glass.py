"""Télécharge les taux de résistance par pays du dashboard OMS GLASS (GLASS-AMR).

Le dashboard (https://worldhealthorg.shinyapps.io/glass-dashboard/) n'a pas d'API : on pilote
la vue « Resistance to antibiotics » avec Playwright et on récupère, pour chaque combinaison
année × type d'infection × bactérie, le CSV « Download data ». Sa section « Data for boxplots »
donne, par pays, le nombre d'infections testées et résistantes.

Chaque fichier est vérifié avant d'être gardé : les filtres écrits en tête du CSV (bactérie,
infection) et l'année du nom de fichier proposé doivent correspondre à la demande. Sans cette
vérification, un téléchargement lancé avant la fin du rafraîchissement Shiny contient les
données de la sélection précédente.

Sortie : data/raw/glass/resistance/*.csv puis data/raw/glass/glass_resistance.csv (fusion).
Usage : python scripts/download_glass.py
"""
from pathlib import Path
import csv
import io

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "glass" / "resistance"
URL = "https://worldhealthorg.shinyapps.io/glass-dashboard/#!/amr"
P = "amr-resistance_antibiotics_year"
INFECTIONS = {"BLOOD": "Bloodstream", "URINE": "Urinary tract", "STOOL": "Gastrointestinal",
              "UROGENITAL": "Gonorrhoea"}


def wait_idle(pg, ms: int = 1500) -> None:
    """Attend que Shiny ait fini de recalculer (classe shiny-busy absente), puis un délai."""
    pg.wait_for_timeout(500)
    pg.wait_for_function("() => !document.documentElement.classList.contains('shiny-busy')", timeout=60000)
    pg.wait_for_timeout(ms)


def options(pg, sel: str) -> list[str]:
    return pg.evaluate(f"() => Object.keys(document.getElementById('{P}-{sel}-select').selectize.options)")


def set_value(pg, sel: str, value: str) -> None:
    pg.evaluate(f"() => document.getElementById('{P}-{sel}-select').selectize.setValue({value!r})")
    wait_idle(pg)


def header_filter(text: str, key: str) -> str:
    for line in text.splitlines()[:8]:
        if key in line:
            return line.split(":", 1)[1].strip().strip('"')
    return ""


def download(pg, dest: Path, year: str, inf: str, pat: str) -> bool:
    for attempt in range(4):
        with pg.expect_download(timeout=90000) as dl:
            pg.evaluate(f"document.getElementById('{P}-dl-data').click()")
        tmp = dest.with_suffix(".tmp")
        dl.value.save_as(tmp)
        txt = tmp.read_text(encoding="utf-8-sig")
        ok = (header_filter(txt, "pathogen filter") == pat
              and header_filter(txt, "Infection type filter") == INFECTIONS[inf]
              and f"in {year}_" in dl.value.suggested_filename)
        if ok:
            tmp.replace(dest)
            return True
        tmp.unlink()
        print(f"  incohérent (essai {attempt + 1}) : {dl.value.suggested_filename} / "
              f"{header_filter(txt, 'pathogen filter')}", flush=True)
        set_value(pg, "pathogen", pat)
        wait_idle(pg, 4000 * (attempt + 1))
    return False


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(accept_downloads=True, viewport={"width": 1500, "height": 1000})
        pg = ctx.new_page()
        pg.goto(URL, timeout=120000)
        pg.wait_for_timeout(20000)
        wait_idle(pg)
        years = sorted(options(pg, "year"))
        print("années", years, flush=True)
        for y in years:
            set_value(pg, "year", y)
            for inf in INFECTIONS:
                set_value(pg, "infsys", inf)
                wait_idle(pg, 2500)
                for pat in options(pg, "pathogen"):
                    dest = OUT / f"{y}_{inf}_{pat.replace(' ', '_').replace('.', '')}.csv"
                    if dest.exists():
                        continue
                    set_value(pg, "pathogen", pat)
                    try:
                        print("ok" if download(pg, dest, y, inf, pat) else "ÉCHEC", dest.name, flush=True)
                    except Exception as e:  # combinaison sans données
                        print("vide", y, inf, pat, str(e)[:60], flush=True)
        b.close()

    rows = []
    for f in sorted(OUT.glob("*.csv")):
        txt = f.read_text(encoding="utf-8-sig")
        if "Data for boxplots" not in txt:
            continue
        part = txt.split("Data for boxplots", 1)[1].strip()
        for r in csv.DictReader(io.StringIO(part)):
            r["Year"] = int(f.name[:4])
            rows.append(r)
    with open(OUT.parent / "glass_resistance.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "lignes pays × bactérie × antibiotique × année")


if __name__ == "__main__":
    main()
