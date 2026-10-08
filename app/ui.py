"""Éléments d'interface partagés par les pages du dashboard (palette, styles, formats)."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data" / "app"

# ---------- palette (validée daltonisme, cf. README) ----------
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e9e6df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
SEQ = ["#f0f5fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

CSS = """
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:ital,wght@0,500;1,500&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stMarkdown, .stText, p, li, label, input, button { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Serif', serif !important; font-weight: 500 !important; letter-spacing: -0.01em; }
code, .mono { font-family: 'IBM Plex Mono', monospace; }
.block-container { padding-top: 4rem; max-width: 1280px; }
.eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: .74rem; letter-spacing: .12em;
           text-transform: uppercase; color: #8a8984; margin-bottom: .2rem; }
.lede { font-size: 1.08rem; color: #52514e; max-width: 62ch; line-height: 1.55; }
.kpi { border-top: 2px solid #0b0b0b; padding-top: .55rem; }
.kpi .v { font-family: 'IBM Plex Serif', serif; font-size: 2.3rem; line-height: 1.05; color: #0b0b0b; }
.kpi .l { font-size: .82rem; color: #52514e; margin-top: .25rem; }
.finding { border-left: 3px solid #2a78d6; padding: .35rem 0 .35rem .9rem; margin: .5rem 0 .9rem 0; }
.finding b { font-weight: 600; }
.finding .src { font-family: 'IBM Plex Mono', monospace; font-size: .72rem; color: #8a8984; }
.note { font-size: .82rem; color: #8a8984; }
.disk { display:inline-block; width:.7rem; height:.7rem; border-radius:50%; margin-right:.35rem; vertical-align:-1px; }
[data-testid="stSidebar"] { background: #f3f1ec; }
.reco { border-left: 6px solid; background: #f3f1ec; padding: .9rem 1.1rem; border-radius: 0 8px 8px 0; margin: .4rem 0 1rem; }
.reco-drug { font-family: 'IBM Plex Serif', serif; font-size: 2rem; line-height: 1.15; color: #0b0b0b; margin: .1rem 0 .3rem; }
.reco-meta { color: #52514e; font-size: .95rem; }
.pill { color: #fff; font-size: .75rem; font-weight: 600; padding: .12rem .5rem; border-radius: 999px; margin-right: .4rem; }
.caveat { font-size: .85rem; color: #52514e; background: #fbf3e4; border: 1px solid #f0dcb4; border-radius: 8px; padding: .7rem .85rem; margin-top: .6rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{name}.parquet")


def style(fig: go.Figure, h: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=h, margin=dict(l=8, r=8, t=82 if legend else 44, b=8),
        title=dict(y=.985, yref="container", yanchor="top", x=0, xanchor="left", pad=dict(l=4)), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="IBM Plex Sans, sans-serif", size=12.5, color=INK2),
        title_font=dict(family="IBM Plex Sans, sans-serif", size=14, color=INK),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK, family="IBM Plex Sans")),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text="",
                    font=dict(color=INK2)),
    )
    if not fig.layout.title.text:
        fig.update_layout(title_text="")
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=INK2))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=INK2))
    return fig


def kpi(col, value: str, label: str) -> None:
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


def fmt(n: float) -> str:
    return f"{n:,.0f}".replace(",", " ")


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f} %".replace(".", ",")


def header(eyebrow: str, title: str, lede: str) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown("# " + title.replace(" ?", " ?").replace(" :", " :"))
    st.markdown(f'<p class="lede">{lede}</p>', unsafe_allow_html=True)
