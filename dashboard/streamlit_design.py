"""JARVIS-style dark-cyber design theme for AccessSentinel Streamlit dashboard.

Usage:
    from streamlit_design import apply_theme, apply_plotly_theme
    apply_theme()   # call after st.set_page_config()
    apply_plotly_theme(fig)  # call before st.plotly_chart()
"""

import streamlit as st

_CSS = """
:root {
    --bg:           #030712;
    --surface:      #0F172A;
    --surface-2:    #1E293B;
    --border:       rgba(255,255,255,0.07);
    --cyan:         #06B6D4;
    --cyan-dim:     rgba(6, 182, 212, 0.15);
    --purple:       #7C3AED;
    --green:        #10B981;
    --amber:        #F59E0B;
    --red:          #EF4444;
    --text:         #F8FAFC;
    --text-muted:   #94A3B8;
}

.stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070D1E 0%, #030712 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stRadio label {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--cyan) !important;
    background: var(--cyan-dim) !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, var(--surface) 0%, rgba(15,23,42,0.6) 100%) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.4rem !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05) !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(6, 182, 212, 0.35) !important;
    box-shadow: 0 0 12px rgba(6, 182, 212, 0.45) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--cyan) 0%, var(--purple) 100%) !important;
    color: #fff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.4rem !important;
    box-shadow: 0 0 14px rgba(6, 182, 212, 0.3) !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.5), 0 0 40px rgba(124, 58, 237, 0.25) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] table { background: var(--surface) !important; color: var(--text) !important; }
[data-testid="stDataFrame"] thead tr { background: var(--surface-2) !important; }
[data-testid="stDataFrame"] thead th {
    color: var(--cyan) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) { background: rgba(30,41,59,0.3) !important; }
[data-testid="stDataFrame"] tbody tr:hover { background: var(--cyan-dim) !important; }

[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 3px !important;
    background: var(--surface) !important;
}
[data-testid="stSpinner"] * { color: var(--cyan) !important; }

hr { border: none !important; border-top: 1px solid var(--border) !important; }

.js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg { background: transparent !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--surface-2); border-radius: 4px; }
"""

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.5)",
    font=dict(family="Inter, sans-serif", color="#94A3B8", size=11),
    colorway=["#06B6D4", "#7C3AED", "#10B981", "#F59E0B", "#EF4444", "#818CF8", "#34D399", "#FB923C"],
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)",
               linecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)",
               linecolor="rgba(255,255,255,0.08)"),
    legend=dict(bgcolor="rgba(15,23,42,0.7)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1),
    margin=dict(l=16, r=16, t=40, b=16),
)


def apply_theme():
    """Inject dark-cyber CSS. Call after st.set_page_config()."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
    st.markdown("<style>#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}</style>", unsafe_allow_html=True)


def apply_plotly_theme(fig):
    """Apply dark-cyber theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_THEME)
    return fig
