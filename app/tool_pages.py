"""Pages de l'outil « Choisir un antibiotique sans laboratoire » (WISCA bayésien sur OMS GLASS)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import wisca as W
from ui import (INK, INK2, MUTED, GRID, SURFACE, BLUE, RED, SEQ, load, style, kpi, fmt, pct, header)

AWARE_COLOR = {"Access": "#1baf7a", "Watch": "#eda100", "Reserve": "#e34948", "?": MUTED}
STEP_COLOR = {"Access": "#1baf7a", "Watch": "#eda100", "Carbapénème": "#eb6834", "Reserve": "#e34948",
              "Aucun": "#52514e", "Pas de données": "#e9e6df"}
SYN_LABEL = {k: v["label"] for k, v in W.SYNDROMES.items()}
PATHO_FR = {"Escherichia coli": "E. coli", "Klebsiella pneumoniae": "K. pneumoniae",
            "Acinetobacter spp.": "Acinetobacter", "Salmonella spp.": "Salmonella",
            "Shigella spp.": "Shigella", "Neisseria gonorrhoeae": "N. gonorrhoeae"}
NOT_MEASURED = {
    "uti": "nitrofurantoïne, fosfomycine, pivmécillinam, gentamicine, amoxicilline-acide clavulanique",
    "bsi_gn": "gentamicine et amikacine pour E. coli et Klebsiella (mesurées seulement pour Acinetobacter), "
              "pipéracilline-tazobactam, associations",
    "diarrhea": "céfixime, ampicilline",
    "gono": "rien d'essentiel : ceftriaxone, céfixime, azithromycine, gentamicine sont suivies",
}
COUNTRY_FR = {"Tunisia": "Tunisie", "Morocco": "Maroc", "Algeria": "Algérie", "France": "France",
              "Egypt": "Égypte", "Lebanon": "Liban", "Jordan": "Jordanie"}


def pts(x: float) -> str:
    """Écart en points de pourcentage : 0,027 -> « 2,7 points »."""
    return f"{100 * x:.1f} points".replace(".", ",")


@st.cache_data
def scored(target: float) -> pd.DataFrame:
    return W.score(load("wisca_coverage"), target)


def country_name(row) -> str:
    return COUNTRY_FR.get(row, row)


# ======================================================================
def page_home() -> None:
    header("Résistome", "Choisir le bon antibiotique quand il n'y a pas d'antibiogramme",
           "Un antibiogramme prend 48 à 72 heures, et beaucoup d'hôpitaux dans le monde n'en font pas. "
           "En attendant, le médecin choisit un antibiotique « probabiliste ». S'il se trompe, le traitement "
           "échoue ; s'il prend par réflexe un antibiotique de dernier recours, il nourrit la résistance de demain. "
           "Résistome estime, pour un pays et une infection, quel antibiotique a le plus de chances de marcher, "
           "en privilégiant ceux qui préservent les molécules de réserve.")
    c = st.columns(4)
    kpi(c[0], "1,14 M", "décès attribuables à la résistance en 2021 (GRAM, The Lancet 2024)")
    kpi(c[1], "1 sur 6", "infections bactériennes résistantes en 2023, 1 sur 3 pour les urinaires (OMS)")
    g = load("glass")
    kpi(c[2], fmt(g.iso3.nunique()), "pays couverts par les données de surveillance de l'OMS")
    dec = load("wisca_decisions")
    held = dec[dec.target == .8]["held"].mean()
    kpi(c[3], pct(held, 1), "des recommandations faites avec 2020-22 encore valables en 2023")
    st.write("")

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown("### La démarche")
        steps = [
            ("1 · Le terrain", "Rapports de surveillance de l'OMS (GLASS) : pour 109 pays, combien d'infections "
             "ont été testées et combien étaient résistantes, par bactérie et par antibiotique (2020-2023)."),
            ("2 · Le calcul", "Pour une infection donnée, la couverture d'un antibiotique est la probabilité "
             "qu'il agisse sur la bactérie, encore inconnue, du patient : on pondère la sensibilité de chaque "
             "bactérie par sa fréquence dans le pays (méthode WISCA), avec un modèle bayésien qui donne un "
             "intervalle d'incertitude et s'appuie sur la région quand le pays a peu de données."),
            ("3 · La décision", "Parmi les antibiotiques qui atteignent la couverture visée, on recommande celui "
             "de spectre le plus étroit, dans l'ordre de la classification AWaRe de l'OMS : Access, puis Watch, "
             "puis carbapénèmes, puis Reserve."),
            ("4 · Le contrôle", "On refait tout avec les seules données 2020-2022 et on confronte aux résultats "
             "2023 : intervalles bien calibrés, recommandations qui tiennent."),
            ("5 · Comprendre", "1,7 million de génomes (NCBI) montrent quels gènes rendent ces bactéries "
             "résistantes et où ils se propagent."),
        ]
        for t, d in steps:
            st.markdown(f'<div class="finding"><b>{t}</b><br>{d}</div>', unsafe_allow_html=True)
    with right:
        st.markdown("### Ce que montrent les données")
        rec = W.recommend(load("wisca_coverage"), .8)
        cov = load("wisca_coverage")
        for syn, txt in [("uti", "infection urinaire"), ("bsi_gn", "bactériémie à Gram négatif")]:
            n_all = cov[cov.syndrome == syn].iso3.nunique()
            r = rec[rec.syndrome == syn].assign(step=lambda x: x.step.map(W.STEP_LABEL))
            carb = (r.step == "Carbapénème").sum()
            res = (r.step == "Reserve").sum()
            none = n_all - len(r)
            st.markdown(
                f'<div class="finding">Pour une <b>{txt}</b>, parmi {n_all} pays, le premier antibiotique '
                f'suivi par l\'OMS qui couvre ≥ 80 % des cas est un <b>carbapénème</b> dans {carb} pays, '
                f'un antibiotique de <b>réserve</b> dans {res}, et <b>aucun</b> n\'y parvient dans {none}.'
                f'<br><span class="src">wisca · OMS GLASS 2021-2023</span></div>', unsafe_allow_html=True)
        tn = rec[rec.iso3 == "TUN"].set_index("syndrome")
        if "bsi_gn" in tn.index:
            r = tn.loc["bsi_gn"]
            st.markdown(
                f'<div class="finding">En <b>Tunisie</b>, face à une bactériémie à Gram négatif, seule la '
                f'<b>{r.antibiotic_fr.lower()}</b> (réserve) atteint 80 % de couverture ({pct(r.q50)}). '
                f'Klebsiella et Acinetobacter, très résistants, y causent près de trois cas sur quatre.'
                f'<br><span class="src">wisca · TUN</span></div>', unsafe_allow_html=True)
        st.markdown('<p class="note">Les données OMS viennent surtout de laboratoires hospitaliers : elles '
                    'surestiment la résistance des infections communautaires simples. Plusieurs antibiotiques '
                    'de première ligne (nitrofurantoïne, fosfomycine, aminosides pour E. coli) ne sont pas '
                    'suivis et ne peuvent pas être évalués.</p>', unsafe_allow_html=True)


# ======================================================================
def page_tool() -> None:
    header("Outil", "Quel antibiotique en probabiliste ?",
           "Choisissez un pays et une infection. Chaque barre donne la probabilité que l'antibiotique agisse "
           "sur la bactérie du patient, avec son intervalle à 95 %. La recommandation est l'antibiotique de "
           "spectre le plus étroit qui atteint la cible.")
    cov = load("wisca_coverage")
    mix = load("wisca_mix")
    countries = cov.drop_duplicates("iso3").assign(name=lambda d: d.country.map(country_name)).sort_values("name")
    names = countries.set_index("iso3")["name"].to_dict()
    c1, c2, c3 = st.columns([1.1, 1.4, 1])
    iso = c1.selectbox("Pays", list(names), index=list(names).index("TUN") if "TUN" in names else 0,
                       format_func=names.get)
    syns = [s for s in W.SYNDROMES if ((cov.iso3 == iso) & (cov.syndrome == s)).any()]
    syn = c2.selectbox("Infection", syns, format_func=SYN_LABEL.get)
    target = c3.slider("Couverture visée", 70, 95, 80, 5, format="%d %%") / 100

    d = scored(target)
    d = d[(d.iso3 == iso) & (d.syndrome == syn)].copy()
    ev = d[d.evaluable].sort_values("q50")
    best = d[d.ok].sort_values(["step", "q50"], ascending=[True, False]).head(1)

    if len(best):
        b = best.iloc[0]
        step = W.STEP_LABEL[b.step]
        col = STEP_COLOR[step]
        st.markdown(
            f'<div class="reco" style="border-color:{col}"><div class="eyebrow">Recommandation · cible '
            f'{pct(target)}</div><div class="reco-drug">{b.antibiotic_fr}</div>'
            f'<div class="reco-meta"><span class="pill" style="background:{col}">{step}</span> '
            f'couverture estimée <b>{pct(b.q50)}</b> (intervalle 95 % : {pct(b.q02)} – {pct(b.q97)}) · '
            f'probabilité d\'atteindre la cible : {pct(b.p_reach)}</div></div>', unsafe_allow_html=True)
        better = ev[(ev.step < b.step)].sort_values("q50", ascending=False).head(1)
        if len(better):
            o = better.iloc[0]
            st.markdown(f'<p class="note">Option plus étroite la mieux placée : {o.antibiotic_fr} '
                        f'({o.aware}), {pct(o.q50)} de couverture, insuffisant pour la cible.</p>',
                        unsafe_allow_html=True)
    else:
        top = ev.sort_values("q50", ascending=False).head(1)
        msg = (f"Le meilleur antibiotique évaluable, {top.iloc[0].antibiotic_fr}, plafonne à "
               f"{pct(top.iloc[0].q50)}." if len(top) else "")
        st.markdown(f'<div class="reco" style="border-color:{RED}"><div class="eyebrow">Recommandation · '
                    f'cible {pct(target)}</div><div class="reco-drug">Aucun antibiotique seul n\'atteint la '
                    f'cible</div><div class="reco-meta">{msg} Dans ce cas, les recommandations cliniques '
                    f'passent par une association ou un avis spécialisé.</div></div>', unsafe_allow_html=True)

    left, right = st.columns([1.5, 1], gap="large")
    with left:
        fig = go.Figure()
        fig.add_vline(x=target, line=dict(color=INK, width=1, dash="dot"))
        for _, r in ev.iterrows():
            colr = AWARE_COLOR.get(r.aware, MUTED)
            fig.add_trace(go.Scatter(x=[r.q02, r.q97], y=[r.antibiotic_fr] * 2, mode="lines",
                                     line=dict(color=colr, width=6), opacity=.35, hoverinfo="skip",
                                     showlegend=False))
            fig.add_trace(go.Scatter(
                x=[r.q50], y=[r.antibiotic_fr], mode="markers",
                marker=dict(size=13, color=colr, line=dict(color=SURFACE, width=2),
                            symbol="diamond" if r.antibiotic in W.CARBAPENEMS else "circle"),
                showlegend=False,
                hovertemplate=(f"<b>{r.antibiotic_fr}</b> · {r.aware}<br>couverture {pct(r.q50)} "
                               f"[{pct(r.q02)} – {pct(r.q97)}]<br>{fmt(r.n_country)} tests dans le pays"
                               + (f"<br>{pct(r.missing_weight)} des cas non documentés (comptés non couverts)"
                                  if r.missing_weight > 0.005 else "") + "<extra></extra>")))
        for a in ["Access", "Watch", "Reserve"]:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=a,
                                     marker=dict(size=11, color=AWARE_COLOR[a])))
        fig.update_xaxes(range=[0, 1.01], tickformat=".0%", title="Probabilité que l'antibiotique agisse")
        fig.update_yaxes(title="", categoryorder="array", categoryarray=ev.antibiotic_fr.tolist())
        fig.update_layout(title=f"{SYN_LABEL[syn]} · {names[iso]} · 2021-2023")
        style(fig, max(320, 34 * len(ev) + 120))
        fig.update_layout(margin=dict(t=44, b=70), legend=dict(yanchor="top", y=-.12, x=0))
        st.plotly_chart(fig, use_container_width=True)
        miss = d[~d.evaluable]
        if len(miss):
            st.markdown('<p class="note">Données insuffisantes (plus de 20 % des cas non documentés) : '
                        + ", ".join(miss.antibiotic_fr) + ".</p>", unsafe_allow_html=True)
        st.markdown(f'<p class="note">Non suivis par l\'OMS, donc non évalués : {NOT_MEASURED[syn]}. '
                    '◆ = carbapénème.</p>', unsafe_allow_html=True)
    with right:
        m = mix[(mix.iso3 == iso) & (mix.syndrome == syn)].sort_values("share")
        fig = go.Figure(go.Bar(x=m.share, y=m.pathogen.map(PATHO_FR), orientation="h",
                               marker=dict(color=BLUE, line=dict(color=SURFACE, width=2)),
                               customdata=m.n_tested,
                               hovertemplate="%{y}<br>%{x:.0%} des cas · %{customdata:,.0f} infections testées<extra></extra>"))
        fig.update_xaxes(tickformat=".0%", title="")
        fig.update_layout(title="Bactéries en cause dans ce pays")
        st.plotly_chart(style(fig, 260, legend=False), use_container_width=True)
        st.markdown(
            '<div class="caveat"><b>Outil d\'aide à la réflexion, pas un avis médical.</b> Il décrit la '
            'résistance moyenne déclarée à l\'OMS, surtout par des hôpitaux : il ne connaît ni le patient '
            '(gravité, allergies, grossesse, fonction rénale, antécédents), ni l\'écologie de votre service. '
            'Les recommandations nationales et l\'antibiogramme, dès qu\'il est disponible, priment.</div>',
            unsafe_allow_html=True)


# ======================================================================
def page_world() -> None:
    header("Carte mondiale", "Où les antibiotiques de première ligne ne suffisent plus",
           "Pour chaque pays : la catégorie du premier antibiotique (le plus étroit) qui atteint la couverture "
           "visée. Plus la carte vire au rouge, plus il faut puiser dans les molécules de dernier recours.")
    c1, c2 = st.columns([1.4, 1])
    syn = c1.selectbox("Infection", list(W.SYNDROMES), format_func=SYN_LABEL.get)
    target = c2.slider("Couverture visée", 70, 95, 80, 5, format="%d %%", key="wt") / 100
    d = scored(target)
    d = d[d.syndrome == syn]
    rec = d[d.ok].sort_values(["step", "q50"], ascending=[True, False]).groupby("iso3").head(1)
    allc = d.drop_duplicates("iso3")[["iso3", "country"]]
    m = allc.merge(rec[["iso3", "antibiotic_fr", "step", "q50"]], on="iso3", how="left")
    m["cat"] = m.step.map(W.STEP_LABEL).fillna("Aucun")
    m["txt"] = np.where(m.antibiotic_fr.notna(), m.antibiotic_fr + " (" + (m.q50 * 100).round().astype("Int64").astype(str) + " %)",
                        "aucun antibiotique évaluable")
    order = ["Access", "Watch", "Carbapénème", "Reserve", "Aucun"]
    fig = go.Figure()
    for cat in order:
        s = m[m.cat == cat]
        if s.empty:
            continue
        fig.add_trace(go.Choropleth(
            locations=s.iso3, z=[1] * len(s), text=s.country, customdata=s.txt, name=f"{cat} ({len(s)})",
            colorscale=[[0, STEP_COLOR[cat]], [1, STEP_COLOR[cat]]], showscale=False, showlegend=True,
            marker_line_color="white", marker_line_width=.5,
            hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>"))
    fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth",
                    bgcolor=SURFACE, landcolor=STEP_COLOR["Pas de données"], showland=True, showcountries=False)
    fig.update_layout(title=f"{SYN_LABEL[syn]} · premier antibiotique atteignant {pct(target)} de couverture")
    st.plotly_chart(style(fig, 520), use_container_width=True)
    counts = m.cat.value_counts().reindex(order).fillna(0).astype(int)
    st.markdown('<p class="note">' + " · ".join(f"{k} : {v} pays" for k, v in counts.items()) +
                ". Gris clair : pas de données OMS pour ce syndrome.</p>", unsafe_allow_html=True)


# ======================================================================
def page_validation() -> None:
    header("Validation", "L'outil aurait-il eu raison en 2023 ?",
           "Tous les réglages ont été choisis en prédisant 2022 avec 2020-2021. Puis on a prédit 2023 avec "
           "2020-2022, sans jamais regarder 2023, et confronté les prédictions à la réalité.")
    v = load("wisca_validation")
    dec = load("wisca_decisions")
    tun = load("wisca_tuning")
    c = st.columns(4)
    kpi(c[0], fmt(len(v)), "couples pays × bactérie × antibiotique testés en 2023")
    kpi(c[1], pct(v.inside.mean(), 1), "des valeurs 2023 dans l'intervalle prédit à 95 %")
    kpi(c[2], pts((v.pred_s - v.obs_s).abs().median()),
        "erreur médiane sur le taux de sensibilité")
    d8 = dec[dec.target == .8]
    kpi(c[3], f"{d8.held.sum()} / {len(d8)}", "recommandations (cible 80 %) encore valables en 2023")
    st.write("")
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        s = v.sample(min(len(v), 2500), random_state=1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRID, width=1),
                                 hoverinfo="skip", showlegend=False))
        for ok, colr, name in [(True, BLUE, "dans l'intervalle"), (False, RED, "hors intervalle")]:
            q = s[s.inside == ok]
            fig.add_trace(go.Scatter(x=q.pred_s, y=q.obs_s, mode="markers", name=name,
                                     marker=dict(size=6, color=colr, opacity=.55),
                                     text=q.iso3 + " · " + q.pathogen + " · " + q.antibiotic,
                                     hovertemplate="%{text}<br>prédit %{x:.0%} · observé %{y:.0%}<extra></extra>"))
        fig.update_xaxes(title="Sensibilité prédite pour 2023", tickformat=".0%", range=[0, 1.02])
        fig.update_yaxes(title="Sensibilité observée en 2023", tickformat=".0%", range=[0, 1.02])
        fig.update_layout(title="Prédit (avec 2020-2022) contre observé (2023)")
        st.plotly_chart(style(fig, 460), use_container_width=True)
    with right:
        p = tun.pivot_table(index="k", columns="tau", values="coverage")
        fig = go.Figure(go.Heatmap(z=p.values, x=[f"{t:g}" for t in p.columns], y=[f"{k:g}" for k in p.index],
                                   colorscale=[[i / 7, c] for i, c in enumerate(SEQ)], zmin=.6, zmax=1,
                                   text=[[pct(x) for x in r] for r in p.values], texttemplate="%{text}",
                                   textfont=dict(size=10), showscale=False,
                                   hovertemplate="K = %{y} · τ = %{x}<br>%{text} dans l'intervalle<extra></extra>"))
        fig.update_xaxes(title="τ : variation d'une année à l'autre (logit)")
        fig.update_yaxes(title="K : poids de l'a priori régional", type="category")
        fig.update_xaxes(type="category")
        fig.update_layout(title="Réglage sur 2022 : part des valeurs dans l'intervalle à 95 %")
        st.plotly_chart(style(fig, 460, legend=False), use_container_width=True)
    st.markdown(
        "**Lecture.** Sans terme de variation annuelle (τ = 0), l'intervalle à 95 % ne contient la valeur de "
        "l'année suivante que 7 fois sur 10 : d'une année à l'autre, ce ne sont pas les mêmes hôpitaux qui "
        "déclarent, la variation dépasse le hasard d'échantillonnage. Avec τ = 0,6 réglé sur 2022, la "
        f"calibration sur 2023 est de {pct(v.inside.mean(), 1)}. Côté décision, une recommandation est jugée "
        "tenue si la couverture observée en 2023 reste au-dessus de la cible moins 5 points.")
    small = v[v.tested_past < 100]
    st.markdown(
        f"**Limite assumée.** Pour les couples peu documentés (< 100 souches passées), l'a priori régional "
        f"ne réduit pas l'erreur ponctuelle (médiane {pts((small.pred_s - small.obs_s).abs().median())} contre "
        f"{pts((small.pred_naive - small.obs_s).abs().median())} avec les seules données du pays) ; son apport "
        "est surtout d'éviter des intervalles faussement étroits.")
