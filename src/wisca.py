"""Choisir un antibiotique sans laboratoire : antibiogramme syndromique pondéré (WISCA) bayésien.

Problème : dans beaucoup de pays, un patient avec une infection urinaire ou une bactériémie reçoit
un antibiotique « probabiliste » avant tout antibiogramme (48-72 h), ou sans antibiogramme du tout.
Le choix se fait souvent par habitude ; quand la résistance locale est élevée, le traitement échoue,
et à l'inverse l'usage réflexe des antibiotiques de réserve accélère la résistance.

Méthode (WISCA, Hebert et al. 2012 ; version bayésienne, Bielicki et al. 2016) :
  couverture(antibiotique | pays, syndrome) = Σ_bactéries  P(bactérie | syndrome, pays)
                                                         × P(sensible | bactérie, antibiotique, pays)
  - P(sensible) : loi Beta a posteriori. Les données du pays (testées / résistantes, OMS GLASS) mettent
    à jour un a priori tiré de la région OMS (puis du monde), d'un poids de K_PRIOR souches : un pays
    avec peu de données est ramené vers sa région au lieu d'afficher un 100 % fragile.
  - P(bactérie) : loi de Dirichlet sur le nombre d'infections testées par bactérie dans le pays.
  - Résistances naturelles (EUCAST) : sensibilité 0 (ex. Klebsiella et ampicilline).
  - Monte-Carlo (N_DRAWS tirages) -> médiane et intervalle crédible à 95 % de la couverture.
  - Recommandation : parmi les antibiotiques dont la couverture atteint la cible avec une probabilité
    ≥ 80 %, on choisit d'abord un « Access » (classification OMS AWaRe), puis « Watch », puis « Reserve ».

Validation : on refait tout avec les seules données 2020-2022 et on confronte aux résultats 2023.

Sorties (data/app/) : glass.parquet, wisca_coverage.parquet, wisca_mix.parquet, wisca_validation*.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
APP = ROOT / "data" / "app"

# Deux paramètres réglés par tune() sur 2020-21 -> 2022 (2023 reste réservée au test) :
K_PRIOR = 5   # poids de l'a priori régional, en « souches équivalentes »
TAU = 0.3     # variation d'une année à l'autre (écart-type sur l'échelle logit) : les hôpitaux qui
              # déclarent changent, la variation dépasse donc le simple hasard d'échantillonnage


def logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def expit(x):
    return 1 / (1 + np.exp(-x))


def year_noise(theta: np.ndarray, rng) -> np.ndarray:
    """Sensibilité de l'année à venir = sensibilité estimée + variation annuelle."""
    return expit(logit(theta) + rng.normal(0, TAU, theta.shape))
N_DRAWS = 4000
RNG = np.random.default_rng(42)

SYNDROMES = {
    "uti": {"label": "Infection urinaire", "specimen": "Urinary tract",
            "pathogens": ["Escherichia coli", "Klebsiella pneumoniae"]},
    "bsi_gn": {"label": "Bactériémie à bacille Gram négatif", "specimen": "Bloodstream",
               "pathogens": ["Escherichia coli", "Klebsiella pneumoniae", "Acinetobacter spp.",
                             "Salmonella spp."]},
    "diarrhea": {"label": "Diarrhée bactérienne invasive (dysenterie)", "specimen": "Gastrointestinal",
                 "pathogens": ["Shigella spp.", "Salmonella spp."]},
    "gono": {"label": "Gonorrhée", "specimen": "Gonorrhoea", "pathogens": ["Neisseria gonorrhoeae"]},
}

# Antibiotiques évalués par syndrome : options utilisées en probabiliste (guide AWaRe de l'OMS, 2022)
# parmi celles que GLASS mesure. Les carbapénèmes ne sont pas proposés contre une dysenterie, etc.
CANDIDATES = {
    "uti": ["Ampicillin", "Co-trimoxazole", "Ciprofloxacin", "Levofloxacin", "Ceftriaxone", "Cefotaxime",
            "Ceftazidime", "Cefepime", "Ertapenem", "Meropenem", "Imipenem", "Colistin"],
    "bsi_gn": ["Ampicillin", "Gentamicin", "Amikacin", "Co-trimoxazole", "Ciprofloxacin", "Ceftriaxone",
               "Cefotaxime", "Ceftazidime", "Cefepime", "Ertapenem", "Meropenem", "Imipenem", "Colistin"],
    "diarrhea": ["Ciprofloxacin", "Azithromycin", "Ceftriaxone", "Cefotaxime"],
    "gono": ["Ceftriaxone", "Cefixime", "Azithromycin", "Gentamicin", "Spectinomycin", "Ciprofloxacin"],
}
CARBAPENEMS = {"Ertapenem", "Meropenem", "Imipenem", "Doripenem"}
MAX_MISSING = 0.2   # au-delà, trop de bactéries non documentées : « données insuffisantes »

# Nom GLASS -> nom de la classification AWaRe 2025
AWARE_NAME = {
    "Co-trimoxazole": "Sulfamethoxazole/trimethoprim",
    "Imipenem": "Imipenem/cilastatin",
    "Amoxicillin-clavulanic acid": "Amoxicillin/clavulanic-acid",
    "Piperacillin-tazobactam": "Piperacillin/tazobactam",
    "Ampicillin-sulbactam": "Ampicillin/sulbactam",
    "Penicillin G": "Benzylpenicillin",
    "Colistin": "Colistin_IV",
    "Minocycline": "Minocycline_IV",
}
FR = {
    "Ampicillin": "Ampicilline", "Amikacin": "Amikacine", "Gentamicin": "Gentamicine",
    "Cefotaxime": "Céfotaxime", "Ceftriaxone": "Ceftriaxone", "Ceftazidime": "Ceftazidime",
    "Cefepime": "Céfépime", "Ciprofloxacin": "Ciprofloxacine", "Levofloxacin": "Lévofloxacine",
    "Co-trimoxazole": "Cotrimoxazole", "Colistin": "Colistine", "Doripenem": "Doripénème",
    "Ertapenem": "Ertapénème", "Imipenem": "Imipénème", "Meropenem": "Méropénème",
    "Tigecycline": "Tigécycline", "Minocycline": "Minocycline", "Tetracycline": "Tétracycline",
    "Nitrofurantoin": "Nitrofurantoïne", "Fosfomycin": "Fosfomycine",
    "Amoxicillin-clavulanic acid": "Amoxicilline-acide clavulanique",
    "Piperacillin-tazobactam": "Pipéracilline-tazobactam", "Cefuroxime": "Céfuroxime",
    "Cefazolin": "Céfazoline", "Ampicillin-sulbactam": "Ampicilline-sulbactam",
    "Azithromycin": "Azithromycine", "Cefixime": "Céfixime", "Spectinomycin": "Spectinomycine",
    "Penicillin G": "Pénicilline G", "Oxacillin": "Oxacilline",
}
# Lignes agrégées par classe dans GLASS : pas des molécules prescriptibles
NOT_DRUGS = {"Third generation cephalosporins resistance", "Carbapenems", "Fluoroquinolones",
             "Aminoglycosides", "Polymyxins", "Methicillin resistance", "Third generation cephalosporins",
             "Fourth generation cephalosporins", "Macrolides", "Penicillins", "Sulfonamides and trimethoprim",
             "Tetracyclines", "Glycylcyclines"}
# Résistances naturelles / inactivité clinique (EUCAST Expected resistant phenotypes v1.2 et notes)
INTRINSIC = {
    "Klebsiella pneumoniae": {"Ampicillin"},
    "Acinetobacter spp.": {"Ampicillin", "Amoxicillin-clavulanic acid", "Cefotaxime", "Ceftriaxone",
                           "Ertapenem", "Cefazolin", "Nitrofurantoin", "Fosfomycin"},
    "Salmonella spp.": {"Gentamicin", "Amikacin", "Cefuroxime", "Cefazolin"},
    "Escherichia coli": set(),
    "Shigella spp.": {"Gentamicin", "Amikacin"},
    "Neisseria gonorrhoeae": set(),
}


def load_glass() -> pd.DataFrame:
    g = pd.read_csv(RAW / "glass" / "glass_resistance.csv")
    g = g.rename(columns={"Specimen": "specimen", "PathogenName": "pathogen", "AntibioticName": "antibiotic",
                          "Iso3": "iso3", "CountryTerritoryArea": "country", "WHORegionName": "region",
                          "InterpretableAST": "tested", "Resistant": "resistant", "Year": "year"})
    g = g[~g["antibiotic"].isin(NOT_DRUGS)]
    g = g[g["tested"] > 0].drop(columns=["ResistancePercentage"], errors="ignore")
    return g


def load_aware() -> dict:
    a = pd.read_excel(RAW / "who" / "aware_2025.xlsx", "AWaRe classification 2025", header=3)
    return dict(zip(a["Antibiotic"].str.strip(), a["Category"].str.strip()))


def aware_of(drug: str, aware: dict) -> str:
    return aware.get(AWARE_NAME.get(drug, drug), "?")


def pooled(g: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return g.groupby(keys, as_index=False)[["tested", "resistant"]].sum()


def coverage_table(g: pd.DataFrame, years: list[int], aware: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Couverture a posteriori (quantiles) par pays × syndrome × antibiotique sur la fenêtre `years`."""
    g = g[g["year"].isin(years)]
    world = pooled(g, ["specimen", "pathogen", "antibiotic"])
    region = pooled(g, ["region", "specimen", "pathogen", "antibiotic"])
    country = pooled(g, ["iso3", "country", "region", "specimen", "pathogen", "antibiotic"])
    qs = np.arange(1, 100)
    cov_rows, mix_rows = [], []
    for key, syn in SYNDROMES.items():
        spec = syn["specimen"]
        measured = set(g.loc[(g["specimen"] == spec) & g["pathogen"].isin(syn["pathogens"]), "antibiotic"])
        drugs = [d for d in CANDIDATES[key] if d in measured]
        cs = country[(country["specimen"] == spec) & country["pathogen"].isin(syn["pathogens"])]
        for (iso3, cname, reg), cdat in cs.groupby(["iso3", "country", "region"]):
            # mélange de bactéries : nb max d'infections testées par bactérie (proxy de l'incidence)
            n_p = np.array([cdat.loc[cdat["pathogen"] == p, "tested"].max() if (cdat["pathogen"] == p).any() else 0
                            for p in syn["pathogens"]], dtype=float)
            if n_p.sum() < 30:
                continue
            w = RNG.dirichlet(n_p + 0.5, N_DRAWS)                     # (draws, pathogens)
            for p, n, wm in zip(syn["pathogens"], n_p, w.mean(0)):
                mix_rows.append({"iso3": iso3, "syndrome": key, "pathogen": p, "n_tested": n, "share": wm})
            for d in drugs:
                theta = np.zeros_like(w)
                n_country = 0
                missing_weight = 0.0
                for j, p in enumerate(syn["pathogens"]):
                    if d in INTRINSIC.get(p, set()):
                        continue                                        # sensibilité 0
                    prior = None
                    r = region[(region["region"] == reg) & (region["specimen"] == spec)
                               & (region["pathogen"] == p) & (region["antibiotic"] == d)]
                    if len(r) and r["tested"].iloc[0] >= 30:
                        prior = 1 - r["resistant"].iloc[0] / r["tested"].iloc[0]
                    else:
                        wd = world[(world["specimen"] == spec) & (world["pathogen"] == p) & (world["antibiotic"] == d)]
                        if len(wd) and wd["tested"].iloc[0] >= 30:
                            prior = 1 - wd["resistant"].iloc[0] / wd["tested"].iloc[0]
                    c = cdat[(cdat["pathogen"] == p) & (cdat["antibiotic"] == d)]
                    t, rr = (c["tested"].iloc[0], c["resistant"].iloc[0]) if len(c) else (0, 0)
                    n_country += t
                    if prior is None and t == 0:
                        # bactérie non documentée pour cet antibiotique : comptée non couverte
                        # (estimation prudente = borne basse), et signalée
                        missing_weight += w[:, j].mean()
                        continue
                    prior = prior if prior is not None else 0.5
                    a0, b0 = K_PRIOR * prior + 0.5, K_PRIOR * (1 - prior) + 0.5
                    theta[:, j] = year_noise(RNG.beta(a0 + (t - rr), b0 + rr, N_DRAWS), RNG)
                cov = (w * theta).sum(1)
                row = {"iso3": iso3, "country": cname, "region": reg, "syndrome": key, "antibiotic": d,
                       "antibiotic_fr": FR.get(d, d), "aware": aware_of(d, aware),
                       "n_country": int(n_country), "missing_weight": missing_weight,
                       "mean": cov.mean()}
                row.update({f"q{q:02d}": v for q, v in zip(qs, np.percentile(cov, qs))})
                cov_rows.append(row)
    return pd.DataFrame(cov_rows), pd.DataFrame(mix_rows)


def p_reach(row: pd.Series, target: float) -> float:
    """P(couverture ≥ cible), lue sur la grille de quantiles q01..q99."""
    qv = row[[f"q{q:02d}" for q in range(1, 100)]].to_numpy(float)
    return float(1 - np.searchsorted(qv, target) / 100)


def stewardship_rank(drug: str, aware: str) -> int:
    """Ordre de préférence : Access, Watch hors carbapénèmes, carbapénèmes, Reserve."""
    if aware == "Access":
        return 0
    if aware == "Watch":
        return 2 if drug in CARBAPENEMS else 1
    return 3 if aware == "Reserve" else 4


STEP_LABEL = {0: "Access", 1: "Watch", 2: "Carbapénème", 3: "Reserve"}


def score(cov: pd.DataFrame, target: float, p_min: float = 0.8) -> pd.DataFrame:
    c = cov.copy()
    c["p_reach"] = c.apply(p_reach, axis=1, target=target)
    c["evaluable"] = c["missing_weight"] <= MAX_MISSING
    c["ok"] = c["evaluable"] & (c["p_reach"] >= p_min)
    c["step"] = [stewardship_rank(d, a) for d, a in zip(c["antibiotic"], c["aware"])]
    return c


def recommend(cov: pd.DataFrame, target: float, p_min: float = 0.8) -> pd.DataFrame:
    """Recommandation par pays × syndrome : l'antibiotique de spectre le plus étroit qui atteint la cible
    (couverture ≥ cible avec une probabilité ≥ p_min) ; à rang égal, la meilleure couverture."""
    c = score(cov, target, p_min)
    return (c[c["ok"]].sort_values(["step", "q50"], ascending=[True, False])
            .groupby(["iso3", "syndrome"]).head(1))


def predict_year(g: pd.DataFrame, train: list[int], test: int, k: float, tau: float) -> pd.DataFrame:
    """Prédit la sensibilité de l'année `test` (pays × bactérie × antibiotique) avec les années `train`."""
    past = g[g["year"].isin(train)]
    fut = g[g["year"] == test]
    region = pooled(past, ["region", "specimen", "pathogen", "antibiotic"])
    world = pooled(past, ["specimen", "pathogen", "antibiotic"])
    country = pooled(past, ["iso3", "region", "specimen", "pathogen", "antibiotic"])
    keys = ["iso3", "region", "specimen", "pathogen", "antibiotic"]
    f = pooled(fut, keys)
    f = f[f["tested"] >= 30].merge(country, on=keys, how="left", suffixes=("", "_past")).fillna(
        {"tested_past": 0, "resistant_past": 0})
    f = f.merge(region.rename(columns={"tested": "t_reg", "resistant": "r_reg"}),
                on=["region", "specimen", "pathogen", "antibiotic"], how="left")
    f = f.merge(world.rename(columns={"tested": "t_w", "resistant": "r_w"}),
                on=["specimen", "pathogen", "antibiotic"], how="left")
    f = f[f["t_w"].notna()]
    prior = np.where(f["t_reg"].fillna(0) >= 30, 1 - f["r_reg"] / f["t_reg"], 1 - f["r_w"] / f["t_w"])
    a = k * prior + 0.5 + (f["tested_past"] - f["resistant_past"])
    b = k * (1 - prior) + 0.5 + f["resistant_past"]
    # intervalle prédictif bêta-binomial pour la proportion observée l'année test
    rng = np.random.default_rng(0)
    th = rng.beta(np.asarray(a)[:, None], np.asarray(b)[:, None], (len(f), 2000))
    th = expit(logit(th) + rng.normal(0, tau, th.shape))
    n = f["tested"].to_numpy()[:, None]
    sim = rng.binomial(n, th) / n
    f["lo"], f["hi"] = np.percentile(sim, 2.5, axis=1), np.percentile(sim, 97.5, axis=1)
    f["obs_s"] = 1 - f["resistant"] / f["tested"]
    f["pred_s"] = a / (a + b)
    f["inside"] = (f["obs_s"] >= f["lo"]) & (f["obs_s"] <= f["hi"])
    f["has_past"] = f["tested_past"] > 0
    f["pred_world"] = 1 - f["r_w"] / f["t_w"]
    f["pred_naive"] = np.where(f["has_past"], 1 - f["resistant_past"] / f["tested_past"].replace(0, np.nan),
                               f["pred_world"])
    return f


def tune(g: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    """Règle K_PRIOR (erreur dans les pays peu documentés) puis TAU (intervalles à 95 % qui
    contiennent ~95 % des valeurs), en prédisant 2022 avec 2020-2021."""
    rows = []
    for k in [0, 2, 5, 10, 20, 50]:
        for tau in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
            f = predict_year(g, [2020, 2021], 2022, k, tau)
            small = f[f["tested_past"] < 100]
            rows.append({"k": k, "tau": tau, "coverage": f["inside"].mean(),
                         "mae": (f["pred_s"] - f["obs_s"]).abs().median(),
                         "mae_small": (small["pred_s"] - small["obs_s"]).abs().median()})
    t = pd.DataFrame(rows)
    k = t.groupby("k")["mae_small"].mean().idxmin()
    sub = t[t["k"] == k]
    tau = sub.loc[(sub["coverage"] - 0.95).abs().idxmin(), "tau"]
    return float(k), float(tau), t


def validate(g: pd.DataFrame, aware: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validation sur 2023, jamais vue : calibration des intervalles et tenue des recommandations."""
    f = predict_year(g, [2020, 2021, 2022], 2023, K_PRIOR, TAU)
    cov_past, _ = coverage_table(g, [2020, 2021, 2022], aware)
    cov_2023, _ = coverage_table(g, [2023], aware)
    dec = []
    for target in (0.8, 0.9):
        rec = recommend(cov_past, target)
        obs = cov_2023[["iso3", "syndrome", "antibiotic", "mean", "n_country"]].rename(
            columns={"mean": "cov_2023", "n_country": "n_2023"})
        m = rec.merge(obs, on=["iso3", "syndrome", "antibiotic"], how="inner")
        m = m[m["n_2023"] >= 50]
        m["target"] = target
        m["held"] = m["cov_2023"] >= target - 0.05      # tolérance de 5 points
        dec.append(m)
    return f, pd.concat(dec, ignore_index=True)


def main() -> None:
    APP.mkdir(parents=True, exist_ok=True)
    g = load_glass()
    aware = load_aware()
    g.to_parquet(APP / "glass.parquet", index=False)
    print(f"GLASS : {len(g):,} lignes, {g['iso3'].nunique()} pays, années {sorted(g['year'].unique())}")
    print("antibiotiques sans catégorie AWaRe :",
          sorted({d for d in g["antibiotic"].unique() if aware_of(d, aware) == "?"}))

    global K_PRIOR, TAU
    K_PRIOR, TAU, tuning = tune(g)
    tuning.to_parquet(APP / "wisca_tuning.parquet", index=False)
    print("réglage (2020-21 -> 2022) :")
    print(tuning.pivot_table(index="k", columns="tau", values="coverage").round(3).to_string())
    print(tuning.groupby("k")[["mae", "mae_small"]].mean().round(4).to_string())
    print(f"retenu : K_PRIOR = {K_PRIOR:g}, TAU = {TAU:g}")

    cov, mix = coverage_table(g, [2021, 2022, 2023], aware)
    cov.to_parquet(APP / "wisca_coverage.parquet", index=False)
    mix.to_parquet(APP / "wisca_mix.parquet", index=False)
    print(f"couvertures : {len(cov):,} (pays × syndrome × antibiotique), {cov['iso3'].nunique()} pays")

    val, dec = validate(g, aware)
    val.to_parquet(APP / "wisca_validation.parquet", index=False)
    dec.to_parquet(APP / "wisca_decisions.parquet", index=False)
    mae = lambda col: (val[col] - val["obs_s"]).abs().median()
    small = val[val["tested_past"] < 100]
    print(f"  pays peu documentés (< 100 souches passées, {len(small)}) : erreur bayésien "
          f"{(small['pred_s'] - small['obs_s']).abs().median():.3f}, naïf "
          f"{(small['pred_naive'] - small['obs_s']).abs().median():.3f}")
    print(f"validation 2023 : {len(val)} couples, intervalle 95 % contient l'observé dans "
          f"{val['inside'].mean():.1%} des cas ; erreur médiane bayésien {mae('pred_s'):.3f}, "
          f"naïf {mae('pred_naive'):.3f}, mondial {mae('pred_world'):.3f}")
    for t, d in dec.groupby("target"):
        print(f"cible {t:.0%} : {len(d)} recommandations confrontées à 2023, tenues dans {d['held'].mean():.1%}")


if __name__ == "__main__":
    main()
