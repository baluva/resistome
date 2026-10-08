"""Résistome : choisir un antibiotique sans antibiogramme (OMS GLASS) et comprendre la résistance (NCBI)."""
from pathlib import Path
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from amr_rules import knowledge_features  # noqa: E402
from ui import (INK, INK2, MUTED, GRID, SURFACE, BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN,  # noqa: E402
                VIOLET, RED, CAT, SEQ, inject_css, load, style, kpi, fmt, pct, header)
import tool_pages as T  # noqa: E402

st.set_page_config(page_title="Résistome", page_icon="🧫", layout="wide")

PHENO = {"R": RED, "I": YELLOW, "S": BLUE}
METHODS = ["Règle catalogue", "Régression logistique", "LightGBM (gènes)", "Hybride"]
METHOD_COLORS = {"Règle catalogue": MUTED, "Régression logistique": ORANGE,
                 "LightGBM (gènes)": AQUA, "Hybride": BLUE}
TOKEN_FR = {"QUINOLONE": "quinolones", "CARBAPENEM": "carbapénèmes", "CEPHALOSPORIN": "céphalosporines",
            "BETA-LACTAM": "bêta-lactamines", "TETRACYCLINE": "tétracyclines", "TIGECYCLINE": "tigécycline",
            "SULFONAMIDE": "sulfamides", "TRIMETHOPRIM": "triméthoprime", "COLISTIN": "colistine",
            "CHLORAMPHENICOL": "chloramphénicol", "AZITHROMYCIN": "azithromycine",
            "GENTAMICIN": "gentamicine", "AMIKACIN": "amikacine", "TOBRAMYCIN": "tobramycine",
            "KANAMYCIN": "kanamycine", "STREPTOMYCIN": "streptomycine", "NITROFURAN": "nitrofuranes",
            "NITROFURANTOIN": "nitrofurantoïne"}


def feat_label(f: str) -> str:
    if f == "k_rule":
        return "▸ règle catalogue positive"
    if f.startswith("k_"):
        t = f[2:]
        return f"▸ nb gènes « {TOKEN_FR.get(t, t.lower())} »"
    return f
SPECIES_ORDER = ["K. pneumoniae", "E. coli", "A. baumannii", "P. aeruginosa", "S. enterica"]
FAMILY_COLORS = {f: CAT[i % 8] for i, f in enumerate([
    "KPC", "NDM", "OXA-48-like", "CTX-M (BLSE)", "VIM", "mcr (colistine)", "IMP",
    "OXA-23/24/58 (Acineto.)", "16S méthylase (aminosides)", "tet(X) (tigécycline)"])}


inject_css()

# ---------- données ----------
@st.cache_resource
def load_models():
    idx = json.loads((MODELS / "index.json").read_text(encoding="utf-8"))
    models = {k: (lgb.Booster(model_file=str(MODELS / v["file"])), v["features"]) for k, v in idx.items()}
    tokens = json.loads((MODELS / "gene_tokens.json").read_text(encoding="utf-8"))
    return models, {sp: {g: set(t) for g, t in d.items()} for sp, d in tokens.items()}


# ======================================================================
def page_overview() -> None:
    header("Génomes · NCBI Pathogen Detection",
           "Lire la résistance aux antibiotiques dans le génome",
           "1,7 million de génomes bactériens séquencés dans le monde, cinq pathogènes classés "
           "prioritaires par l'OMS. Où circulent les gènes de résistance les plus dangereux, "
           "et peut-on prédire l'antibiogramme d'une souche sans la cultiver ?")
    k = load("kpis")
    ct = load("country_totals")
    metrics = load("metrics")
    c = st.columns(4)
    kpi(c[0], fmt(k.n_genomes.sum()), "génomes analysés (5 espèces)")
    kpi(c[1], fmt(ct.iso3.nunique()), "pays d'origine cartographiés")
    kpi(c[2], fmt(k.n_ast.sum()), "souches avec antibiogramme de labo")
    n_pairs = metrics[["species", "antibiotic"]].drop_duplicates().shape[0]
    kpi(c[3], fmt(n_pairs), "modèles espèce × antibiotique")
    st.write("")

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown("### Ce que disent les données")
        for f in findings():
            st.markdown(f'<div class="finding">{f[0]}<br><span class="src">{f[1]}</span></div>',
                        unsafe_allow_html=True)
    with right:
        cs = load("class_share")
        keep = ["BETA-LACTAM", "AMINOGLYCOSIDE", "QUINOLONE", "SULFONAMIDE", "TETRACYCLINE",
                "TRIMETHOPRIM", "PHENICOL", "MACROLIDE", "COLISTIN", "FOSFOMYCIN"]
        labels = {"BETA-LACTAM": "Bêta-lactamines", "AMINOGLYCOSIDE": "Aminosides",
                  "QUINOLONE": "Quinolones", "SULFONAMIDE": "Sulfamides", "TETRACYCLINE": "Tétracyclines",
                  "TRIMETHOPRIM": "Triméthoprime", "PHENICOL": "Phénicolés", "MACROLIDE": "Macrolides",
                  "COLISTIN": "Colistine", "FOSFOMYCIN": "Fosfomycine"}
        m = cs[cs["class"].isin(keep)].pivot_table(index="class", columns="species", values="share")
        m = m.reindex(index=keep, columns=SPECIES_ORDER).fillna(0)
        m.index = [labels[i] for i in m.index]
        fig = go.Figure(go.Heatmap(
            z=m.values, x=m.columns, y=m.index, colorscale=[[i / 7, c] for i, c in enumerate(SEQ)],
            zmin=0, zmax=1, xgap=2, ygap=2,
            text=[[pct(v) for v in row] for row in m.values], texttemplate="%{text}",
            textfont=dict(size=11),
            hovertemplate="%{x} · %{y}<br>%{text} des génomes portent ≥1 déterminant<extra></extra>",
            colorbar=dict(title="", tickformat=".0%", thickness=10, len=.8)))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="Part des génomes portant au moins un déterminant, par classe")
        st.plotly_chart(style(fig, 430, legend=False), use_container_width=True)
        st.markdown('<p class="note">Inclut les gènes intrinsèques (ex. blaEC chez E. coli, '
                    'OXA-51 chez A. baumannii), d\'où des parts proches de 100 % pour les bêta-lactamines.</p>',
                    unsafe_allow_html=True)

    st.markdown("### Multirésistance : combien de classes d'antibiotiques un génome neutralise-t-il ?")
    mdr = load("mdr")
    mdr["share"] = mdr["n"] / mdr.groupby("species")["n"].transform("sum")
    mdr["bucket"] = pd.cut(mdr["n_classes"], [-1, 0, 2, 5, 10],
                           labels=["0 classe", "1–2 classes", "3–5 classes (MDR)", "6+ classes"])
    b = mdr.groupby(["species", "bucket"], observed=True)["share"].sum().reset_index()
    fig = px.bar(b, y="species", x="share", color="bucket", orientation="h",
                 category_orders={"species": SPECIES_ORDER[::-1],
                                  "bucket": ["0 classe", "1–2 classes", "3–5 classes (MDR)", "6+ classes"]},
                 color_discrete_sequence=["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    fig.update_traces(marker_line_color=SURFACE, marker_line_width=2,
                      hovertemplate="%{y} · %{fullData.name}<br>%{x:.1%} des génomes<extra></extra>")
    fig.update_xaxes(tickformat=".0%", title="")
    fig.update_yaxes(title="")
    fig.update_layout(title="Part des génomes selon le nombre de classes d'antibiotiques couvertes")
    st.plotly_chart(style(fig, 300), use_container_width=True)
    st.markdown('<p class="note">Classes portées par des gènes acquis (hors mutations ponctuelles et pompes '
                'd\'efflux). MDR = résistance à ≥ 3 classes, définition ECDC/CDC.</p>', unsafe_allow_html=True)


def findings() -> list[tuple[str, str]]:
    fy = load("family_year")
    yt = load("year_totals")
    fr = load("family_reservoir")
    met = load("metrics")
    out = []

    def share(sp, fam, y0, y1):
        s = fy[(fy.species == sp) & (fy.watch_family == fam) & fy.year.between(y0, y1)]["n_pos"].sum()
        n = yt[(yt.species == sp) & yt.year.between(y0, y1)]["n"].sum()
        return s / n if n else np.nan

    a, b = share("K. pneumoniae", "NDM", 2010, 2014), share("K. pneumoniae", "NDM", 2021, 2025)
    out.append((f"Chez <b>K. pneumoniae</b>, la carbapénémase <b>NDM</b> passe de {pct(a, 1)} des génomes "
                f"séquencés (2010-14) à <b>{pct(b, 1)}</b> (2021-25).", "family_year · NCBI PD"))
    a, b = share("K. pneumoniae", "KPC", 2010, 2014), share("K. pneumoniae", "KPC", 2021, 2025)
    out.append((f"Sur la même période, <b>KPC</b> évolue de {pct(a, 1)} à {pct(b, 1)} : "
                f"le paysage des carbapénémases se redistribue.", "family_year · NCBI PD"))
    m = fr[(fr.species == "E. coli") & (fr.watch_family == "mcr (colistine)")].set_index("reservoir")["share"]
    if "Porc" in m and "Humain" in m:
        out.append((f"Le gène <b>mcr</b> (résistance à la colistine, antibiotique de dernier recours) est "
                    f"<b>{m['Porc'] / m['Humain']:.0f}×</b> plus fréquent dans les E. coli d'origine porcine "
                    f"({pct(m['Porc'], 1)}) qu'humaine ({pct(m['Humain'], 1)}).", "family_reservoir · One Health"))
    hy = met[met.method == "Hybride"]
    good = (hy.bal_acc >= .9).mean()
    out.append((f"Le génome seul prédit l'antibiogramme avec une exactitude équilibrée ≥ 90 % pour "
                f"<b>{pct(good)}</b> des {len(hy)} couples espèce × antibiotique testés "
                f"(validation sur des études jamais vues).", "metrics · modèle hybride, CV groupée"))
    piv = met.pivot_table(index=["species", "antibiotic"], columns="method", values="bal_acc").dropna()
    w_lgb = (piv["Hybride"] > piv["LightGBM (gènes)"] + .005).sum()
    l_lgb = (piv["Hybride"] < piv["LightGBM (gènes)"] - .005).sum()
    w_r = (piv["Hybride"] > piv["Règle catalogue"] + .005).sum()
    l_r = (piv["Hybride"] < piv["Règle catalogue"] - .005).sum()
    me = met.pivot_table(index=["species", "antibiotic"], columns="method", values="me").dropna().median()
    out.append((f"Donner au modèle la <b>classe de chaque gène</b> (connaissance experte) l'améliore "
                f"sur {w_lgb} couples contre {l_lgb}. Face à la règle experte seule, l'hybride fait "
                f"jeu égal ({w_r} victoires, {l_r} défaites) mais lance moins de fausses alertes "
                f"(ME {pct(me['Hybride'], 1)} contre {pct(me['Règle catalogue'], 1)}).",
                "metrics · couples où la règle existe"))
    q = load("quality")
    bad = q[q.quarantine]
    if len(bad):
        r = bad.iloc[0]
        out.append((f"Audit qualité : l'étude <b>{r.bioproject}</b> déclare {pct(r.kpn_ampicillin_pct_S)} "
                    f"de K. pneumoniae sensibles à l'ampicilline, une espèce naturellement résistante. "
                    f"Ses {fmt(r.n_isolates)} souches sont mises en quarantaine.",
                    "quality · concordance génotype/phénotype"))
    return out


# ======================================================================
def page_surveillance() -> None:
    header("Surveillance génomique", "Où circulent les gènes qui inquiètent",
           "Part des génomes séquencés portant chaque famille de gènes sous surveillance. "
           "Ce sont des génomes déposés, pas un échantillon représentatif : on lit des tendances "
           "et des contrastes, pas des taux d'incidence.")
    fy, yt = load("family_year"), load("year_totals")
    fc, ft = load("family_country"), load("country_totals")
    c1, c2 = st.columns([1, 2])
    sp = c1.selectbox("Espèce", SPECIES_ORDER, index=0)
    fams = sorted(fy[fy.species == sp].groupby("watch_family")["n_pos"].sum()
                  .loc[lambda s: s >= 50].sort_values(ascending=False).index.tolist(),
                  key=lambda f: list(FAMILY_COLORS).index(f) if f in FAMILY_COLORS else 99)
    default = [f for f in ["KPC", "NDM", "OXA-48-like", "OXA-23/24/58 (Acineto.)", "CTX-M (BLSE)",
                           "mcr (colistine)"] if f in fams][:3]
    sel = c2.multiselect("Familles de gènes (3 max. pour rester lisible)", fams, default=default,
                         max_selections=4)

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        fam_map = st.selectbox("Carte : famille affichée", sel or fams, key="mapfam")
        d = ft[(ft.species == sp) & (ft.n >= 50)].merge(
            fc[(fc.species == sp) & (fc.watch_family == fam_map)][["country", "share", "n_pos"]],
            on="country", how="left").fillna({"share": 0, "n_pos": 0})
        d = d[d.iso3.notna()]
        fig = go.Figure(go.Choropleth(
            locations=d.iso3, z=d.share, text=d.country,
            customdata=np.c_[d.n_pos, d.n],
            colorscale=[[i / 7, c] for i, c in enumerate(SEQ)], zmin=0,
            zmax=max(d.share.quantile(.95), .01), marker_line_color="white", marker_line_width=.5,
            colorbar=dict(tickformat=".0%", thickness=10, len=.7, title=""),
            hovertemplate="<b>%{text}</b><br>%{z:.1%} des génomes<br>%{customdata[0]:,} / %{customdata[1]:,}<extra></extra>"))
        fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth",
                        bgcolor=SURFACE, landcolor="#efede8", showland=True, showcountries=False)
        fig.update_layout(title=f"{fam_map} chez {sp} : part des génomes par pays (≥ 50 génomes)")
        st.plotly_chart(style(fig, 430, legend=False), use_container_width=True)
    with right:
        d = fy[(fy.species == sp) & fy.watch_family.isin(sel)]
        d = d[d.n >= 100]
        fig = go.Figure()
        for f in sel:
            s = d[d.watch_family == f].sort_values("year")
            fig.add_trace(go.Scatter(x=s.year, y=s.share, name=f, mode="lines+markers",
                                     line=dict(width=2, color=FAMILY_COLORS.get(f, INK2)),
                                     marker=dict(size=7, line=dict(color=SURFACE, width=2)),
                                     hovertemplate=f"{f} · %{{x}}<br>%{{y:.1%}} des génomes<extra></extra>"))
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        fig.update_layout(title="Évolution 2005-2025 (années avec ≥ 100 génomes)", hovermode="x unified")
        st.plotly_chart(style(fig, 430), use_container_width=True)

    st.markdown("### One Health : la même résistance chez l'humain, l'animal et dans l'environnement")
    fr, rt = load("family_reservoir"), load("reservoir_totals")
    res_order = ["Humain", "Volaille", "Porc", "Bovin", "Animal de compagnie", "Aliment / autre", "Environnement"]
    d = fr[(fr.species == sp) & fr.watch_family.isin(sel) & fr.reservoir.isin(res_order)]
    d = d.merge(rt[(rt.species == sp)][["reservoir", "n"]].rename(columns={"n": "n_res"}), on="reservoir")
    d = d[d.n_res >= 100]
    if d.empty:
        st.info("Pas assez de génomes non humains pour cette espèce.")
    else:
        fig = px.bar(d, x="reservoir", y="share", color="watch_family", barmode="group",
                     category_orders={"reservoir": res_order, "watch_family": sel},
                     color_discrete_map=FAMILY_COLORS, custom_data=["n_pos", "n_res"])
        fig.update_traces(marker_line_color=SURFACE, marker_line_width=2,
                          hovertemplate="%{x} · %{fullData.name}<br>%{y:.1%} (%{customdata[0]:,} / %{customdata[1]:,})<extra></extra>")
        fig.update_yaxes(tickformat=".0%", title="")
        fig.update_xaxes(title="")
        fig.update_layout(title=f"{sp} : part des génomes porteurs selon le réservoir (≥ 100 génomes)")
        st.plotly_chart(style(fig, 360), use_container_width=True)
        counts = rt[(rt.species == sp) & rt.reservoir.isin(res_order)].set_index("reservoir")["n"]
        st.markdown('<p class="note">Effectifs : ' + " · ".join(
            f"{r} {fmt(counts.get(r, 0))}" for r in res_order if counts.get(r, 0) >= 100) + "</p>",
            unsafe_allow_html=True)


# ======================================================================
def page_model() -> None:
    header("Génotype → phénotype", "Le génome peut-il remplacer l'antibiogramme ?",
           "Pour chaque souche, on connaît les déterminants détectés par AMRFinderPlus et le résultat "
           "du labo (S / I / R). Quatre approches sont comparées, testées sur des études (BioProjects) "
           "absentes de l'apprentissage.")
    met = load("metrics")
    imp = load("importances")
    sp = st.selectbox("Espèce", [s for s in SPECIES_ORDER if s in met.species.unique()])
    d = met[met.species == sp]
    order = (d[d.method == "Hybride"].sort_values("bal_acc")["antibiotic"]).tolist()

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        fig = go.Figure()
        piv = d.pivot_table(index="antibiotic", columns="method", values="bal_acc").reindex(order)
        for a in order:  # tige entre la règle (ou LR) et le meilleur score
            row = piv.loc[a].dropna()
            fig.add_trace(go.Scatter(x=[row.min(), row.max()], y=[a, a], mode="lines",
                                     line=dict(color=GRID, width=2), hoverinfo="skip", showlegend=False))
        for m in METHODS:
            s = d[d.method == m]
            fig.add_trace(go.Scatter(
                x=s.bal_acc, y=s.antibiotic, mode="markers", name=m,
                marker=dict(size=11 if m == "Hybride" else 8, color=METHOD_COLORS[m],
                            line=dict(color=SURFACE, width=2)),
                customdata=np.c_[s.n, s.vme, s.me],
                hovertemplate=f"<b>%{{y}}</b> · {m}<br>exactitude équilibrée %{{x:.3f}}<br>"
                              "VME %{customdata[1]:.1%} · ME %{customdata[2]:.1%}<br>n = %{customdata[0]:,}<extra></extra>"))
        fig.update_xaxes(title="Exactitude équilibrée", range=[.4, 1.01])
        fig.update_yaxes(categoryorder="array", categoryarray=order, title="")
        fig.update_layout(title=f"{sp} : performance par antibiotique")
        style(fig, max(380, 26 * len(order) + 140))
        fig.update_layout(margin=dict(t=44, b=70), legend=dict(yanchor="top", y=-.08, x=0))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        lg = d[d.method == "Hybride"]
        fig = go.Figure()
        fig.add_shape(type="rect", x0=0, x1=.03, y0=0, y1=.015, fillcolor="#d7ebdf",
                      line=dict(width=0), layer="below")
        fig.add_trace(go.Scatter(x=lg.me, y=lg.vme, mode="markers", text=lg.antibiotic,
                                 marker=dict(size=9, color=BLUE, line=dict(color=SURFACE, width=2)),
                                 hovertemplate="<b>%{text}</b><br>ME %{x:.1%} · VME %{y:.1%}<extra></extra>",
                                 showlegend=False))
        worst = lg.nlargest(3, "vme")
        for _, r in worst.iterrows():
            fig.add_annotation(x=r.me, y=r.vme, text=r.antibiotic, showarrow=False, yshift=12,
                               font=dict(size=11, color=INK2))
        fig.update_xaxes(title="ME : souche sensible prédite résistante", tickformat=".0%", rangemode="tozero")
        fig.update_yaxes(title="VME : souche résistante prédite sensible", tickformat=".0%", rangemode="tozero")
        fig.update_layout(title="Erreurs cliniques du modèle hybride")
        st.plotly_chart(style(fig, 400, legend=False), use_container_width=True)
        st.markdown('<p class="note">Zone verte en bas à gauche : seuils FDA (ME ≤ 3 %, VME ≤ 1,5 %). La VME est l\'erreur la plus grave : elle ferait prescrire un '
                    'antibiotique inefficace. Les seuils FDA s\'appliquent aux tests de sensibilité '
                    'commerciaux ; ils servent ici de repère.</p>', unsafe_allow_html=True)

    st.markdown("### Quels déterminants le modèle utilise-t-il ?")
    drug = st.selectbox("Antibiotique", order[::-1], index=0)
    di = imp[(imp.species == sp) & (imp.antibiotic == drug)].head(12).iloc[::-1].copy()
    di["label"] = di.gene.map(feat_label)
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        di = di.assign(direction=np.where(di.shap_if_present > 0, "pousse vers R", "pousse vers S"))
        fig = px.bar(di, x="mean_abs_shap", y="label", orientation="h", color="direction",
                     color_discrete_map={"pousse vers R": RED, "pousse vers S": BLUE},
                     custom_data=["prevalence", "pct_r_if_present", "subclass"])
        fig.update_traces(marker_line_color=SURFACE, marker_line_width=2,
                          hovertemplate="<b>%{y}</b> (%{customdata[2]})<br>|SHAP| moyen %{x:.3f}<br>"
                                        "présent chez %{customdata[0]:.1%} des souches<br>"
                                        "%{customdata[1]:.1%} R quand présent<extra></extra>")
        fig.update_xaxes(title="Contribution moyenne |SHAP|")
        fig.update_yaxes(title="", categoryorder="array", categoryarray=di.label.tolist())
        fig.update_layout(title=f"{sp} · {drug}")
        st.plotly_chart(style(fig, 420), use_container_width=True)
    with c2:
        row = d[d.antibiotic == drug].set_index("method")
        st.markdown(f"**{fmt(row.n.iloc[0])} souches**, {pct(row.pct_r.iloc[0])} résistantes, "
                    f"{int(row.n_projects.iloc[0])} études · validation {row.cv.iloc[0]}")
        show = row.reindex([m for m in METHODS if m in row.index])[["bal_acc", "auc", "vme", "me"]].rename(columns={
            "bal_acc": "Exactitude équil.", "auc": "AUC", "vme": "VME", "me": "ME"}).astype(float)
        st.dataframe(show.style.format({"Exactitude équil.": "{:.3f}", "AUC": "{:.3f}",
                                        "VME": "{:.1%}", "ME": "{:.1%}"}, na_rep="—"),
                     use_container_width=True)
        top = imp[(imp.species == sp) & (imp.antibiotic == drug)].head(3)
        st.markdown("Variables principales : " + ", ".join(f"`{feat_label(g)}`" for g in top.gene))
        st.markdown('<p class="note">Rouge : la présence du déterminant augmente la probabilité '
                    'de résistance. Les variables ▸ sont les connaissances ajoutées au modèle hybride. Une mutation notée <code>gène_X123Y</code> est une substitution '
                    'ponctuelle dans une protéine chromosomique.</p>', unsafe_allow_html=True)


# ======================================================================
def page_predict() -> None:
    header("Antibiogramme in silico", "Une souche réelle, son génome, son antibiogramme",
           "Choisissez une souche du jeu de test : le modèle lit ses déterminants de résistance et "
           "prédit chaque résultat, comparé ici au résultat réel du laboratoire. Ajoutez ou retirez "
           "un gène pour voir l'effet.")
    models, gene_tokens = load_models()
    demo = load("demo_isolates")
    oof = load("oof")
    c1, c2 = st.columns([1, 2])
    sp = c1.selectbox("Espèce", [s for s in SPECIES_ORDER if s in demo.species.unique()])
    d = demo[(demo.species == sp) & demo.isolate.isin(oof.isolate)].copy()  # hors quarantaine
    d["label"] = d.isolate + " · " + d.country.fillna("pays ?") + " · " + d.year.astype("string").fillna("?")
    # souches intéressantes en tête : beaucoup de gènes (profil multirésistant)
    d["n_genes"] = d.genes.str.count(",") + 1
    # en tête : souches au profil mixte (R et S) avec beaucoup de résultats de labo
    lab_stats = oof[oof.species == sp].groupby("isolate")["y"].agg(["size", "mean"])
    d = d.join(lab_stats, on="isolate")
    d["mixed"] = d["mean"].between(.25, .75) & (d["size"] >= 10) & d.country.notna()
    d = d.sort_values(["mixed", "size", "n_genes"], ascending=False)
    pick = c2.selectbox("Souche (profils mixtes R/S en premier)", d.label.tolist()[:400])
    row = d[d.label == pick].iloc[0]
    base_genes = row.genes.split(",")
    drugs = {k.split("|")[1]: v for k, v in models.items() if k.split("|")[0] == sp}
    all_feats = sorted(f for f in set().union(*[set(f) for _, f in drugs.values()]) if not f.startswith("k_"))
    genes = st.multiselect("Déterminants détectés (modifiable)",
                           sorted(set(all_feats) | set(base_genes)), default=base_genes)
    st.markdown(f'<p class="note">{row.reservoir} · {row.country or "pays non renseigné"} · '
                f'{row.year if pd.notna(row.year) else "année ?"} · {len(base_genes)} déterminants '
                f'dans le génome d\'origine</p>', unsafe_allow_html=True)

    lab = oof[(oof.isolate == row.isolate)].set_index("antibiotic")["y"]
    rows = []
    for drug, (bst, feats) in sorted(drugs.items()):
        known = gene_tokens.get(sp, {})
        kf = knowledge_features([known[g] for g in genes if g in known], drug)
        x = pd.DataFrame([[kf.get(f, 0) if f.startswith("k_") else int(f in genes) for f in feats]],
                         columns=feats)
        p = float(bst.predict(x)[0])
        truth = lab.get(drug)
        rows.append({"antibiotic": drug, "p": p, "pred": "R" if p >= .5 else "S",
                     "lab": None if truth is None or pd.isna(truth) else ("R" if truth == 1 else "S")})
    res = pd.DataFrame(rows).sort_values("p", ascending=True)
    tested = res.dropna(subset=["lab"])
    agree = (tested.pred == tested.lab).mean() if len(tested) else np.nan

    c = st.columns(3)
    kpi(c[0], f"{(res.pred == 'R').sum()} / {len(res)}", "antibiotiques prédits inefficaces (R)")
    kpi(c[1], fmt(len(tested)), "résultats de labo disponibles")
    kpi(c[2], pct(agree) if len(tested) else "—", "concordance prédiction / labo")
    st.write("")
    fig = go.Figure()
    fig.add_vline(x=.5, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_trace(go.Bar(x=res.p, y=res.antibiotic, orientation="h",
                         marker=dict(color=[PHENO[p] for p in res.pred], line=dict(color=SURFACE, width=2)),
                         hovertemplate="<b>%{y}</b><br>P(résistant) = %{x:.2f}<extra></extra>",
                         showlegend=False))
    t = res.dropna(subset=["lab"])
    fig.add_trace(go.Scatter(x=[1.04] * len(t), y=t.antibiotic, mode="markers+text",
                             marker=dict(size=13, color=[PHENO[v] for v in t.lab], symbol="circle",
                                         line=dict(color=INK, width=1)),
                             text=t.lab, textposition="middle right", textfont=dict(color=INK, size=11),
                             name="Résultat labo", hovertemplate="<b>%{y}</b><br>labo : %{text}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%", title="Probabilité prédite de résistance  ·  pastille = résultat labo")
    fig.update_yaxes(title="")
    fig.update_layout(title=f"Antibiogramme prédit · {row.isolate}")
    st.plotly_chart(style(fig, max(360, 24 * len(res) + 100), legend=False), use_container_width=True)
    st.markdown(f'<p class="note"><span class="disk" style="background:{RED}"></span>R résistant '
                f'<span class="disk" style="background:{BLUE};margin-left:.8rem"></span>S sensible. '
                'Les modèles affichés sont entraînés sur toutes les souches ; la concordance hors '
                'échantillon est celle de la page « Génotype → phénotype ».</p>', unsafe_allow_html=True)


# ======================================================================
def page_method() -> None:
    header("Méthode", "Sources, choix et limites", "Tout est reproductible depuis les fichiers publics du NCBI.")
    st.markdown("""
**Sources : outil de choix d'antibiotique**
- [OMS GLASS](https://worldhealthorg.shinyapps.io/glass-dashboard/) (Global Antimicrobial Resistance and Use
  Surveillance System) : par pays, bactérie, antibiotique et année (2020-2023), nombre d'infections testées et
  résistantes ; 109 pays. Récupéré depuis le dashboard officiel (`scripts/download_glass.py`), chaque fichier
  vérifié (bactérie, infection, année).
- [Classification AWaRe 2025 de l'OMS](https://iris.who.int/handle/10665/382244) : Access / Watch / Reserve.
- Résistances naturelles : EUCAST, *Expected resistant phenotypes*.
- Méthode WISCA : Hebert et al., *Infect Control Hosp Epidemiol* 2012 ; version bayésienne : Bielicki et al.,
  *J Antimicrob Chemother* 2016.
- Contexte : GRAM, *The Lancet* 2024 (1,14 million de décès attribuables en 2021) ; OMS, *Global antibiotic
  resistance surveillance report 2025* (1 infection sur 6 résistante en 2023).

**Sources : partie génomique**
- [NCBI Pathogen Detection](https://www.ncbi.nlm.nih.gov/pathogens/) : métadonnées AMR des isolats
  (`latest_snps/AMR/*.amr.metadata.tsv`) pour *Klebsiella*, *E. coli/Shigella*, *Salmonella*,
  *Acinetobacter*, *P. aeruginosa*. Déterminants détectés par **AMRFinderPlus** ; antibiogrammes
  (AST) déposés via NCBI Antibiogram par 391 études : FDA NARMS (Salmonella, E. coli), Walter Reed
  Army Institute of Research, Agence canadienne d'inspection des aliments, laboratoires de santé
  publique GenomeTrakr, Brigham & Women's Hospital, Université de Liverpool…
- Fichiers publiés le 7 oct. 2026 (versions PDG000000012.2544, PDG000000004.6347, PDG000000002.4252,
  PDG000000010.1971, PDG000000036.1807) ; base AMRFinderPlus 2026-08-07.1.
- Catalogue de référence AMRFinderPlus (`ReferenceGeneCatalog.txt`, `AMRProt-mutation.tsv`)
  pour rattacher chaque gène à une classe d'antibiotique.

**Pipeline** : `ingest.py` (1,7 Go de TSV → Parquet) → `aggregate.py` (DuckDB, agrégats légers
versionnés) → `model.py` (scikit-learn, LightGBM, SHAP) → ce dashboard Streamlit.

**Choix de modélisation**
- Variables : présence/absence de chaque gène ou mutation (≥ 5 occurrences). Les hits AMRFinderPlus
  incertains (`PARTIAL_END_OF_CONTIG`, `MISTRANSLATION`, `HMM`, `INTERNAL_STOP`) sont écartés.
- Cible : R contre S (les résultats intermédiaires sont exclus de l'apprentissage).
- **Validation croisée groupée par BioProject** : une étude entière est soit en apprentissage soit
  en test. Les souches d'une même étude sont souvent clonales ; une validation aléatoire
  surestimerait les scores.
- **Modèle hybride** : aux gènes s'ajoutent des variables de connaissance (nombre de déterminants
  acquis par classe d'antibiotique, verdict de la règle). Sans elles, un modèle purement statistique
  ne reconnaît pas un allèle jamais vu en apprentissage (ex. `gyrA_D87N`, présent dans une seule étude).
- Référence « règle catalogue » : R si un déterminant de la bonne classe est présent, **en excluant
  les gènes intrinsèques** (blaEC, OXA-51, blaADC, blaPDC, OXA-50, fosA, oqxAB…), sinon la règle
  prédit R pour toute l'espèce.

**Limites**
- Les génomes déposés ne sont pas un échantillon représentatif : surreprésentation des souches
  résistantes, des épidémies et des pays qui séquencent beaucoup (États-Unis, Royaume-Uni).
- Les antibiogrammes mélangent standards (CLSI / EUCAST) et années de seuils différents : bruit
  d'étiquette, surtout près des seuils (céfépime, amikacine).
- Certaines résistances ne sont pas « lisibles » en présence/absence : surexpression de pompes
  d'efflux, nombre de copies d'un gène, perte de porine partielle.
""")
    st.markdown("### Audit qualité : une étude aux phénotypes incohérents")
    q = load("quality")
    q = q[q.n_isolates >= 20].sort_values("concordance").reset_index(drop=True)
    bad = q[q.quarantine]
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        fig = go.Figure(go.Scatter(
            x=q.concordance, y=q.n_tests, mode="markers", text=q.bioproject,
            marker=dict(size=9, color=[RED if b else BLUE for b in q.quarantine],
                        line=dict(color=SURFACE, width=2)),
            customdata=q.n_isolates,
            hovertemplate="<b>%{text}</b><br>concordance %{x:.0%}<br>%{y:,} tests · %{customdata} souches<extra></extra>"))
        fig.add_vline(x=.5, line=dict(color=MUTED, dash="dot", width=1))
        for _, r in bad.iterrows():
            fig.add_annotation(x=r.concordance, y=r.n_tests, text=f"{r.bioproject} · quarantaine",
                               showarrow=False, xanchor="left", xshift=10, font=dict(size=11, color=INK))
        fig.update_xaxes(title="Concordance règle catalogue / antibiogramme labo", tickformat=".0%", range=[0, 1.02])
        fig.update_yaxes(title="Nombre de tests (échelle log)", type="log")
        fig.update_layout(title="Concordance génotype / phénotype par étude (≥ 20 souches)")
        st.plotly_chart(style(fig, 380, legend=False), use_container_width=True)
    with c2:
        if len(bad):
            r = bad.iloc[0]
            kpn = r.kpn_ampicillin_pct_S
            st.markdown(
                f"Dans **{r.bioproject}** ({fmt(r.n_isolates)} souches, {fmt(r.n_tests)} tests), "
                f"le génotype n'explique le phénotype que dans **{pct(r.concordance)}** des cas, "
                f"contre {pct(q.concordance.median())} en médiane pour les autres études.")
            if pd.notna(kpn):
                st.markdown(
                    f"Contrôle biologique : **{pct(kpn)}** des *K. pneumoniae* y sont notées "
                    "**sensibles à l'ampicilline**. C'est impossible : l'espèce y est naturellement "
                    "résistante (pénicillinase chromosomique SHV). Les phénotypes ont très "
                    "probablement été attribués aux mauvaises souches lors du dépôt.")
            st.markdown("Règle appliquée : quarantaine si concordance < 50 % sur ≥ 100 tests. "
                        "Une seule étude est concernée ; elle est exclue de l'apprentissage et de l'évaluation.")

    met = load("metrics")
    st.markdown("**Résumé des performances (médianes sur les couples où la règle existe)**")
    both = met[met.antibiotic.isin(met[met.method == "Règle catalogue"].antibiotic)]
    keys = both[both.method == "Règle catalogue"][["species", "antibiotic"]]
    both = both.merge(keys, on=["species", "antibiotic"])
    s = both.groupby("method")[["bal_acc", "f1", "vme", "me"]].median().reindex(
        METHODS).rename(columns={"bal_acc": "Exactitude équilibrée", "f1": "F1", "vme": "VME", "me": "ME"})
    s.index.name = "Méthode"
    st.dataframe(s.style.format({"Exactitude équilibrée": "{:.3f}", "F1": "{:.3f}", "VME": "{:.1%}", "ME": "{:.1%}"}),
                 use_container_width=True)


pg = st.navigation({
    "Choisir un antibiotique": [
        st.Page(T.page_home, title="Le problème", default=True, url_path="accueil"),
        st.Page(T.page_tool, title="Outil : quel antibiotique ?", url_path="outil"),
        st.Page(T.page_world, title="Carte mondiale", url_path="carte"),
        st.Page(T.page_validation, title="Validation 2023", url_path="validation"),
    ],
    "Comprendre : les gènes de résistance": [
        st.Page(page_overview, title="Vue d'ensemble des génomes", url_path="genomes"),
        st.Page(page_surveillance, title="Surveillance génomique", url_path="surveillance"),
        st.Page(page_model, title="Génotype → phénotype", url_path="modele"),
        st.Page(page_predict, title="Antibiogramme in silico", url_path="prediction"),
    ],
    "Méthode": [
        st.Page(page_method, title="Sources, méthode & limites", url_path="methode"),
    ],
})
with st.sidebar:
    st.markdown('<div class="eyebrow">Résistome</div>', unsafe_allow_html=True)
    st.markdown('<p class="note">Données OMS GLASS 2020-2023 et NCBI Pathogen Detection '
                '(oct. 2026). Projet de Louey et Sara Barbirou.</p>', unsafe_allow_html=True)
pg.run()
