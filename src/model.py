"""Génotype -> phénotype : peut-on prédire l'antibiogramme à partir du génome ?

Pour chaque couple (espèce, antibiotique) assez documenté, on compare trois approches,
évaluées en validation croisée groupée par BioProject (une étude entière est soit en
apprentissage, soit en test : pas de fuite entre souches clonales d'une même étude) :

  1. Règle "catalogue"   : R si le génome porte un déterminant AMRFinderPlus dont la classe
                           couvre l'antibiotique (l'approche bio-informatique classique)
  2. Régression logistique sur la présence/absence des déterminants
  3. LightGBM sur les mêmes variables (capte les combinaisons : porine + BLSE, etc.)
  4. Hybride : LightGBM + variables "connaissances" (nb de déterminants acquis par classe
     d'antibiotique + verdict de la règle). Généralise aux allèles jamais vus en apprentissage.

Métriques cliniques : VME (very major error : souche R prédite S, le pire cas pour le
patient) et ME (major error : souche S prédite R). Repères FDA pour un test de sensibilité :
VME <= 1,5 % et ME <= 3 %.

Sorties : data/app/metrics.parquet, importances.parquet, oof.parquet, models/*.txt
"""
from pathlib import Path
import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
APP = ROOT / "data" / "app"
MODELS = ROOT / "models"

MIN_N, MIN_MINORITY, MIN_FEAT_FREQ = 300, 25, 5

from amr_rules import RULES, INTRINSIC_FAMILIES, KNOW_TOKENS, rule_hit, knowledge_features


def oxa_family_map() -> dict:
    """blaOXA-66 -> 'OXA-51 family' etc. (lu dans le catalogue AMRFinderPlus)."""
    cat = pd.read_csv(ROOT / "data" / "raw" / "ReferenceGeneCatalog.txt", sep="	", dtype=str)
    fam = cat["product_name"].str.extract(r"(OXA-\d+ family)")[0]
    return dict(zip(cat["allele"][fam.notna()], fam[fam.notna()]))


def tokens(row) -> set:
    return set(str(row["class"]).split("/")) | set(str(row["subclass"]).split("/"))


def acquired_tokens(genes: pd.DataFrame, isolates: pd.Index) -> pd.Series:
    """isolate -> liste des jeux de jetons de ses déterminants acquis (hors intrinsèques et efflux)."""
    g = genes[(genes["class"] != "EFFLUX") & ~genes["intrinsic"]]
    lst = g.groupby("isolate")["tok"].agg(list)
    return lst.reindex(isolates).apply(lambda x: x if isinstance(x, list) else [])


def knowledge_matrix(toks: pd.Series, drug: str) -> pd.DataFrame:
    return pd.DataFrame([knowledge_features(t, drug) for t in toks], index=toks.index)


def clinical_metrics(y, pred, prob=None) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    r, s = y == 1, y == 0
    return {
        "bal_acc": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "auc": roc_auc_score(y, prob) if prob is not None else np.nan,
        "vme": float(((pred == 0) & r).sum() / max(r.sum(), 1)),
        "me": float(((pred == 1) & s).sum() / max(s.sum(), 1)),
    }


def lgb_model():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                              min_child_samples=10, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, class_weight="balanced", verbose=-1)


def main() -> None:
    APP.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    iso = pd.read_parquet(PROC / "isolates.parquet")
    iso = iso[iso["has_ast"]].set_index("isolate")
    ast = pd.read_parquet(PROC / "ast.parquet")
    genes = pd.read_parquet(PROC / "genes.parquet")
    genes = genes[genes["isolate"].isin(iso.index)].copy()
    genes["tok"] = genes.apply(tokens, axis=1)
    oxa = oxa_family_map()
    sp_of = iso["species"]
    genes["intrinsic"] = [
        (g in INTRINSIC_FAMILIES[sp_of[i]] or fam in INTRINSIC_FAMILIES[sp_of[i]]
         or oxa.get(g) in INTRINSIC_FAMILIES[sp_of[i]]) and not pt
        for i, g, fam, pt in zip(genes["isolate"], genes["gene"], genes["gene_family"], genes["is_point"])
    ]
    gene_meta = genes.drop_duplicates("gene").set_index("gene")[["class", "subclass", "is_point"]]

    # ---- Audit qualité : concordance génotype/phénotype par étude ----
    # Une étude où la règle catalogue est d'accord avec le labo moins d'une fois sur deux
    # (pire que le hasard) signale des phénotypes mal attribués : quarantaine.
    toks_all = acquired_tokens(genes, iso.index)
    a_r = ast[ast["antibiotic"].isin(RULES) & (ast["phenotype"] != "I")].copy()
    a_r["rule"] = [rule_hit(d, set().union(*toks_all[i]) if toks_all[i] else set())
                   for i, d in zip(a_r["isolate"], a_r["antibiotic"])]
    a_r["ok"] = a_r["rule"] == (a_r["phenotype"] == "R").astype(int)
    a_r["bioproject"] = iso["bioproject"].reindex(a_r["isolate"]).values
    a_r["species"] = iso["species"].reindex(a_r["isolate"]).values
    audit = a_r.groupby("bioproject").agg(n_tests=("ok", "size"), n_isolates=("isolate", "nunique"),
                                          concordance=("ok", "mean")).reset_index()
    audit["quarantine"] = (audit["concordance"] < 0.5) & (audit["n_tests"] >= 100)
    # contrôle biologique : K. pneumoniae est naturellement résistante à l'ampicilline (blaSHV)
    kp = a_r[(a_r["species"] == "K. pneumoniae") & (a_r["antibiotic"] == "ampicillin")]
    kp_s = kp.groupby("bioproject")["phenotype"].apply(lambda p: (p == "S").mean())
    audit["kpn_ampicillin_pct_S"] = audit["bioproject"].map(kp_s)
    audit.sort_values("concordance").to_parquet(APP / "quality.parquet", index=False)
    bad = set(audit.loc[audit["quarantine"], "bioproject"])
    print(f"Quarantaine : {sorted(bad)} ({iso['bioproject'].isin(bad).sum()} souches)", flush=True)
    iso = iso[~iso["bioproject"].isin(bad)]

    rows, imps, oofs, feature_lists = [], [], [], {}
    for sp, iso_sp in iso.groupby("species"):
        g_sp = genes[genes["isolate"].isin(iso_sp.index)]
        X_all = pd.crosstab(g_sp["isolate"], g_sp["gene"]).clip(upper=1)
        X_all = X_all.reindex(iso_sp.index, fill_value=0)
        X_all = X_all.loc[:, X_all.sum() >= MIN_FEAT_FREQ]
        X_all.columns = [str(c) for c in X_all.columns]
        toks_sp = acquired_tokens(g_sp, iso_sp.index)
        a_sp = ast[ast["isolate"].isin(iso_sp.index) & (ast["phenotype"] != "I")]
        for drug, a in a_sp.groupby("antibiotic"):
            y = (a.set_index("isolate")["phenotype"] == "R").astype(int)
            if len(y) < MIN_N or min(y.sum(), (1 - y).sum()) < MIN_MINORITY:
                continue
            X = X_all.loc[y.index]
            X = X.loc[:, X.sum() > 0]
            K = knowledge_matrix(toks_sp.loc[y.index], drug)
            K = K.loc[:, K.sum() > 0]
            XK = pd.concat([X, K], axis=1)  # hybride : gènes + connaissances
            groups = iso_sp.loc[y.index, "bioproject"].fillna("NA").to_numpy()
            # Validation groupée par étude ; repli stratifié si trop peu d'études
            # ou si une étude concentre l'essentiel d'une classe.
            splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                          .split(X, y, groups))
            cv_type = "groupée (BioProject)"
            if any(len(tr) == 0 or len(te) == 0 or y.iloc[tr].nunique() < 2 for tr, te in splits):
                splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
                cv_type = "stratifiée"
            p_lr, p_gb, p_hy = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
            for tr, te in splits:
                lr = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")
                lr.fit(X.iloc[tr], y.iloc[tr])
                p_lr[te] = lr.predict_proba(X.iloc[te])[:, 1]
                gb = lgb_model().fit(X.iloc[tr], y.iloc[tr])
                p_gb[te] = gb.predict_proba(X.iloc[te])[:, 1]
                hy = lgb_model().fit(XK.iloc[tr], y.iloc[tr])
                p_hy[te] = hy.predict_proba(XK.iloc[te])[:, 1]

            base = {"species": sp, "antibiotic": drug, "n": len(y), "pct_r": y.mean(),
                    "n_projects": len(set(groups)), "cv": cv_type}
            if drug in RULES:
                rule = K["k_rule"].reindex(y.index).fillna(0).to_numpy(int) if "k_rule" in K else np.zeros(len(y), int)
                rows.append({**base, "method": "Règle catalogue", **clinical_metrics(y, rule)})
            rows.append({**base, "method": "Régression logistique",
                         **clinical_metrics(y, (p_lr >= .5).astype(int), p_lr)})
            rows.append({**base, "method": "LightGBM (gènes)",
                         **clinical_metrics(y, (p_gb >= .5).astype(int), p_gb)})
            rows.append({**base, "method": "Hybride",
                         **clinical_metrics(y, (p_hy >= .5).astype(int), p_hy)})
            oofs.append(pd.DataFrame({"isolate": y.index, "species": sp, "antibiotic": drug,
                                      "y": y.values, "p_gb": p_gb, "p_hy": p_hy,
                                      "rule": rule if drug in RULES else -1}))

            # modèle final + SHAP (contribution moyenne de chaque déterminant)
            X = XK  # le modèle livré est l'hybride
            final = lgb_model().fit(X, y)
            key = f"{sp}|{drug}"
            fname = key.replace(". ", "_").replace(" ", "_").replace("|", "__").replace("/", "-")
            final.booster_.save_model(MODELS / f"{fname}.txt")
            feature_lists[key] = {"file": f"{fname}.txt", "features": list(X.columns)}
            sv = shap.TreeExplainer(final).shap_values(X)
            sv = sv[1] if isinstance(sv, list) else sv
            imp = pd.DataFrame({"gene": X.columns, "mean_abs_shap": np.abs(sv).mean(0),
                                "shap_if_present": [sv[X[c].to_numpy() >= 1, i].mean()
                                                    for i, c in enumerate(X.columns)],
                                "prevalence": (X >= 1).mean().values,
                                "pct_r_if_present": [y[X[c] >= 1].mean() for c in X.columns]})
            imp = imp.sort_values("mean_abs_shap", ascending=False).head(15)
            imp.insert(0, "antibiotic", drug); imp.insert(0, "species", sp)
            imps.append(imp)
            m = rows[-1]
            print(f"{sp:14s} {drug:32s} n={len(y):5d}  Hybride bal_acc={m['bal_acc']:.3f} "
                  f"VME={m['vme']:.3f} ME={m['me']:.3f}  [{cv_type}]", flush=True)

    metrics = pd.DataFrame(rows)
    importances = pd.concat(imps).merge(gene_meta, left_on="gene", right_index=True, how="left")
    metrics.to_parquet(APP / "metrics.parquet", index=False)
    importances.to_parquet(APP / "importances.parquet", index=False)
    pd.concat(oofs).to_parquet(APP / "oof.parquet", index=False)
    acq = genes[(genes["class"] != "EFFLUX") & ~genes["intrinsic"]]
    gene_tokens = {sp: {g: sorted(t) for g, t in zip(d["gene"], d["tok"])}
                   for sp, d in acq.assign(species=sp_of.reindex(acq["isolate"]).values)
                   .drop_duplicates(["species", "gene"]).groupby("species")}
    (MODELS / "gene_tokens.json").write_text(json.dumps(gene_tokens), encoding="utf-8")
    (MODELS / "index.json").write_text(json.dumps(feature_lists, indent=1), encoding="utf-8")

    piv = metrics.pivot_table(index="method", values=["bal_acc", "vme", "me", "auc"], aggfunc="median")
    print("\nMédianes sur", metrics[["species", "antibiotic"]].drop_duplicates().shape[0], "couples :")
    print(piv.round(3))


if __name__ == "__main__":
    main()
