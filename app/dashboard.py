import streamlit as st
import pandas as pd
import json
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import xgboost as xgb

# ══════════════════════════════════════════════════════════════════════════════
# THEME — edit this section to change the entire dashboard palette
# ══════════════════════════════════════════════════════════════════════════════
THEME = {
    "bg_app":         "#DCEDFF",
    "bg_sidebar":     "#FFFFFF",
    "bg_card":        "#FEFFFE",
    "bg_nav_active":  "#2B2D42",
    "text_primary":   "#2B2D42",
    "text_secondary": "#0B3954",
    "text_muted":     "#8A8FA8",
    "border":         "#F0F1F3",
    "border_sidebar": "#E8EAED",
    "accent":         "#4F8AEF",
    "accent_light":   "#EEF0FF",
    "danger":         "#EF4444",
    "danger_light":   "#FEE2E2",
    "success":        "#10B981",
    "success_light":  "#D1FAE5",
    "warning":        "#D97706",
    "warning_light":  "#FEF3C7",
    "radius":         "20px",
    "radius_sm":      "10px",
    "font":           "Inter",
}

# ── Navigation options — defined once, used everywhere ───────────────────────
NAV_OVERVIEW   = "📊  Overview"
NAV_SCORING    = "🔎  Transaction Scoring"
NAV_MONITORING = "📡  Model Monitoring"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: '{THEME["font"]}', sans-serif;
}}
.stApp {{
    background-color: {THEME["bg_app"]};
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {THEME["bg_sidebar"]};
    border-right: 1px solid {THEME["border_sidebar"]};
}}
[data-testid="stSidebar"] > div:first-child {{
    padding-top: 1.5rem;
}}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ display: none; }}

/* ── Main content ── */
.main .block-container {{
    padding: 2rem 2.5rem;
    max-width: 1400px;
}}

/* ── KPI Cards ── */
.kpi-card {{
    background: {THEME["bg_card"]};
    border-radius: {THEME["radius"]};
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    border: 1px solid {THEME["border"]};
    height: 100%;
}}
.kpi-label {{
    font-size: 11px;
    font-weight: 500;
    color: {THEME["text_muted"]};
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 8px;
}}
.kpi-value {{
    font-size: 26px;
    font-weight: 700;
    color: {THEME["text_primary"]};
    line-height: 1;
    margin-bottom: 6px;
}}
.kpi-sub {{
    font-size: 12px;
    color: {THEME["text_muted"]};
}}

/* ── Section titles ── */
.section-title {{
    font-size: 14px;
    font-weight: 600;
    color: {THEME["text_primary"]};
    margin: 0 0 0.75rem 0;
    letter-spacing: -0.01em;
}}
.page-title {{
    font-size: 22px;
    font-weight: 700;
    color: {THEME["text_primary"]};
    letter-spacing: -0.02em;
    margin-bottom: 2px;
}}
.page-subtitle {{
    font-size: 13px;
    color: {THEME["text_muted"]};
    margin-bottom: 1.5rem;
}}

/* ── Sidebar navigation radio ── */
[data-testid="stSidebar"] .stRadio > div {{
    gap: 4px;
    padding: 0 8px;
}}
[data-testid="stSidebar"] .stRadio label {{
    display: flex;
    align-items: center;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 400;
    color: {THEME["text_secondary"]};
    cursor: pointer;
    width: 100%;
    transition: background 0.12s;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    background: #F3F4F6;
    color: {THEME["text_primary"]};
}}
[data-testid="stSidebar"] .stRadio [data-checked="true"] > label {{
    background: #E5E7EB;
    color: {THEME["text_primary"]};
    font-weight: 500;
}}
[data-testid="stSidebar"] .stRadio [type="radio"] {{
    display: none;
}}
[data-testid="stSidebar"] .stRadio label > div:first-child {{
    display: none;
}}

/* ── Badges ── */
.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.02em;
}}
.badge-legit {{ background: {THEME["success_light"]}; color: {THEME["success"]}; }}
.badge-warn  {{ background: {THEME["warning_light"]}; color: {THEME["warning"]}; }}
.badge-fraud {{ background: {THEME["danger_light"]};  color: {THEME["danger"]};  }}

/* ── Monitoring table ── */
.monitoring-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
.monitoring-table th {{
    background: {THEME["bg_app"]};
    color: {THEME["text_muted"]};
    font-weight: 500;
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.07em;
    padding: 10px 14px;
    border-bottom: 1px solid {THEME["border_sidebar"]};
    text-align: left;
}}
.monitoring-table td {{
    padding: 10px 14px;
    border-bottom: 1px solid {THEME["border"]};
    color: {THEME["text_primary"]};
}}
.monitoring-table tr:last-child td {{ border-bottom: none; }}

/* ── Divider ── */
.divider {{
    height: 1px;
    background: {THEME["border"]};
    margin: 1.25rem 0;
}}

/* ── Sidebar model info row ── */
.info-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 14px;
}}
.info-label {{
    font-size: 12px;
    color: {THEME["text_secondary"]};
}}
.info-value {{
    font-size: 12px;
    font-weight: 600;
    color: {THEME["text_primary"]};
}}

/* ── Plotly charts rounded corners ── */
[data-testid="stPlotlyChart"] > div {{
    border-radius: {THEME["radius"]} !important;
    overflow: hidden;
}}
[data-testid="stPlotlyChart"] iframe {{
    border-radius: {THEME["radius"]} !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: #D1D5DB; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

# ── Plotly helpers ─────────────────────────────────────────────────────────────
def base_layout(height=300, **kwargs):
    return dict(
        paper_bgcolor=THEME["bg_card"],
        plot_bgcolor=THEME["bg_card"],
        font=dict(family=THEME["font"], size=12, color=THEME["text_secondary"]),
        margin=dict(l=10, r=10, t=16, b=10),
        height=height,
        showlegend=False,
        **kwargs
    )

def styled_axes(fig, xformat=None, yformat=None, xrange=None, yrange=None):
    xargs = dict(gridcolor=THEME["border"], linecolor=THEME["border"], zeroline=False)
    yargs = dict(gridcolor=THEME["border"], linecolor=THEME["border"], zeroline=False)
    if xformat: xargs["tickformat"] = xformat
    if yformat: yargs["tickformat"] = yformat
    if xrange:  xargs["range"] = xrange
    if yrange:  yargs["range"] = yrange
    fig.update_xaxes(**xargs)
    fig.update_yaxes(**yargs)
    return fig

# ── Assets (everything the app needs lives in app/assets) ─────────────────────
ASSETS = Path(__file__).parent / "assets"

@st.cache_resource
def load_model():
    m = xgb.XGBClassifier()
    m.load_model(ASSETS / "xgb_fraud_model.json")
    return m

@st.cache_data
def load_overview():
    return pd.read_parquet(ASSETS / "overview.parquet")

@st.cache_data
def load_test_sample():
    return pd.read_parquet(ASSETS / "test_sample.parquet")

@st.cache_data
def load_config():
    with open(ASSETS / "selected_features.json") as f:
        features = json.load(f)
    with open(ASSETS / "optimal_threshold.json") as f:
        threshold = json.load(f)["optimal_threshold"]
    return features, threshold

model                       = load_model()
overview                    = load_overview()
test_sample                 = load_test_sample()
features, optimal_threshold = load_config()

X_test   = test_sample[features]
y_test_s = test_sample["isFraud"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    tab = st.radio(
        "Navigation",
        [NAV_OVERVIEW, NAV_SCORING, NAV_MONITORING],
        label_visibility="collapsed"
    )

    st.markdown(f"""
    <div class="divider"></div>
    <div style="padding:0 14px;margin-bottom:10px;">
        <div style="font-size:10px;font-weight:500;color:{THEME['text_muted']};
                    text-transform:uppercase;letter-spacing:0.07em;">
            Model Info
        </div>
    </div>
    """, unsafe_allow_html=True)

    for label, value in [
        ("ROC-AUC",   "0.9103"),
        ("PR-AUC",    "0.5303"),
        ("Threshold", str(optimal_threshold)),
        ("Features",  str(len(features))),
    ]:
        st.markdown(f"""
        <div class="info-row">
            <span class="info-label">{label}</span>
            <span class="info-value">{value}</span>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
if tab == NAV_OVERVIEW:
    st.markdown('<div class="page-title">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Real e-commerce transactions · '
        'Vesta Corporation · ~6 months</div>',
        unsafe_allow_html=True
    )

    n_total  = len(overview)
    n_fraud  = int(overview["isFraud"].sum())
    fraud_rt = overview["isFraud"].mean() * 100

    c1, c2, c3, c4 = st.columns(4, gap="small")
    for col, label, value, sub in [
        (c1, "Total Transactions", f"{n_total:,}",       "train + val + test"),
        (c2, "Fraud Rate",         f"{fraud_rt:.2f}%",   f"{n_fraud:,} fraud cases"),
        (c3, "Raw Features",       "434",                "393 transaction + 41 identity"),
        (c4, "Selected Features",  str(len(features)),   "SHAP CV + correlation filter"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        st.markdown('<div class="section-title">Transaction Amount by Class</div>',
                    unsafe_allow_html=True)
        sample = overview.sample(5000, random_state=42)
        fig = px.box(
            sample, x="isFraud", y="TransactionAmt",
            color="isFraud",
            color_discrete_map={0: THEME["accent"], 1: THEME["danger"]},
            labels={"isFraud": "", "TransactionAmt": "Amount ($)"},
        )
        fig.update_xaxes(tickvals=[0, 1], ticktext=["Legitimate", "Fraud"])
        fig.update_layout(**base_layout(height=300))
        styled_axes(fig, yrange=[0, 1000])
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title">Fraud Rate by Product Category</div>',
                    unsafe_allow_html=True)
        fprod = (
            overview.groupby("ProductCD")["isFraud"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "fraud_rate", "count": "n"})
            .sort_values("fraud_rate")
        )
        fig = px.bar(
            fprod, x="fraud_rate", y="ProductCD", orientation="h",
            color="fraud_rate",
            color_continuous_scale=[THEME["accent_light"], THEME["accent"]],
            labels={"fraud_rate": "Fraud Rate", "ProductCD": ""},
        )
        fig.update_layout(**base_layout(height=300), coloraxis_showscale=False)
        styled_axes(fig, xformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Fraud Rate Over Time</div>',
                unsafe_allow_html=True)
    daily = (
        overview.assign(day_int=overview["day"].astype(int))
        .groupby("day_int")["isFraud"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"day_int": "day", "mean": "fraud_rate", "count": "n"})
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["day"], y=daily["fraud_rate"],
        mode="lines",
        line=dict(color=THEME["accent"], width=2),
        fill="tozeroy",
        fillcolor="rgba(79,138,239,0.08)",
    ))
    fig.add_vline(x=25, line_dash="dash", line_color=THEME["danger"], line_width=1,
                  annotation_text="Volume spike", annotation_font_size=11,
                  annotation_font_color=THEME["text_muted"])
    fig.update_layout(**base_layout(height=200))
    styled_axes(fig, yformat=".1%")
    fig.update_yaxes(title="Fraud Rate")
    fig.update_xaxes(title="Day since reference point")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Transaction Scoring
# ══════════════════════════════════════════════════════════════════════════════
elif tab == NAV_SCORING:
    st.markdown('<div class="page-title">Transaction Scoring</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Select a transaction from the test sample '
        'to score and explain</div>',
        unsafe_allow_html=True
    )

    test_indices = X_test.index.tolist()
    selected_idx = st.selectbox(
        "Transaction",
        test_indices,
        format_func=lambda x: (
            f"#{x}  —  FRAUD" if y_test_s.loc[x] == 1
            else f"#{x}  —  LEGITIMATE"
        )
    )

    row        = X_test.loc[[selected_idx]]
    true_label = int(y_test_s.loc[selected_idx])
    proba      = float(model.predict_proba(row)[0, 1])
    is_fraud   = proba >= optimal_threshold
    decision   = "FRAUD" if is_fraud else "LEGITIMATE"
    correct    = is_fraud == bool(true_label)

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="small")
    for col, label, value, sub, color_key in [
        (c1, "Fraud Score", f"{proba:.4f}", "Model output probability",
         THEME["danger"] if is_fraud else THEME["accent"]),
        (c2, "Decision", decision, f"Threshold: {optimal_threshold}",
         THEME["danger"] if is_fraud else THEME["success"]),
        (c3, "True Label", "FRAUD" if true_label else "LEGITIMATE", "Ground truth",
         THEME["danger"] if true_label else THEME["success"]),
        (c4, "Correct?", "Yes" if correct else "No", "Model vs. ground truth",
         THEME["success"] if correct else THEME["danger"]),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color_key};font-size:22px;">
                    {value}
                </div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    col_gauge, col_shap = st.columns([1, 1.6], gap="medium")

    with col_gauge:
        st.markdown('<div class="section-title">Fraud Probability</div>',
                    unsafe_allow_html=True)
        bar_color = THEME["danger"] if is_fraud else THEME["accent"]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba, 4),
            number={"font": {"size": 34, "family": THEME["font"],
                             "color": THEME["text_primary"]},
                    "valueformat": ".4f"},
            gauge={
                "axis": {"range": [0, 1],
                         "tickfont": {"size": 10, "color": THEME["text_muted"]}},
                "bar": {"color": bar_color, "thickness": 0.22},
                "bgcolor": THEME["bg_app"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, optimal_threshold], "color": THEME["accent_light"]},
                    {"range": [optimal_threshold, 1], "color": THEME["danger_light"]}
                ],
                "threshold": {
                    "line": {"color": THEME["text_primary"], "width": 2},
                    "thickness": 0.75,
                    "value": optimal_threshold
                }
            }
        ))
        fig.update_layout(
            height=260,
            paper_bgcolor=THEME["bg_card"],
            margin=dict(l=20, r=20, t=30, b=0),
            font=dict(family=THEME["font"])
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f'<div style="text-align:center;font-size:11px;'
            f'color:{THEME["text_muted"]};margin-top:-8px;">'
            f'Threshold at <strong>{optimal_threshold}</strong> · '
            f'Red zone = fraud flag</div>',
            unsafe_allow_html=True
        )

    with col_shap:
        st.markdown('<div class="section-title">Feature Impact — Why this decision?</div>',
                    unsafe_allow_html=True)
        with st.spinner("Computing SHAP explanation..."):
            explainer   = shap.TreeExplainer(model)
            shap_vals   = explainer.shap_values(row)
            shap_series = (
                pd.Series(shap_vals[0], index=features)
                .sort_values(key=abs, ascending=False)
                .head(12)
            )
            bar_colors = [
                THEME["danger"] if v > 0 else THEME["accent"]
                for v in shap_series.values
            ]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            fig.patch.set_facecolor(THEME["bg_card"])
            ax.set_facecolor(THEME["bg_card"])
            ax.barh(shap_series.index[::-1], shap_series.values[::-1],
                    color=bar_colors[::-1], height=0.55)
            ax.axvline(0, color=THEME["border_sidebar"], linewidth=1)
            ax.set_xlabel("SHAP value  (→ toward fraud  |  ← toward legitimate)",
                          fontsize=10, color=THEME["text_muted"])
            ax.tick_params(labelsize=10, colors=THEME["text_secondary"])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(axis="x", color=THEME["border"], linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Monitoring
# ══════════════════════════════════════════════════════════════════════════════
elif tab == NAV_MONITORING:
    st.markdown('<div class="page-title">Model Monitoring</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Performance degradation across temporal '
        'production windows · Evidently AI</div>',
        unsafe_allow_html=True
    )

    monitor_rows = [
        ("Reference (training)", 0.982, 0.945, 0.262, "3.52%", "12.7%", "Healthy"),
        ("Window 1 · month 1",   0.923, 0.817, 0.190, "3.41%", "14.7%", "Watch"),
        ("Window 2 · month 2",   0.910, 0.800, 0.176, "3.49%", "15.8%", "Watch"),
    ]
    rows_html = ""
    for window, roc, rec, prec, fr, dr, status in monitor_rows:
        badge = "badge-legit" if status == "Healthy" else "badge-warn"
        rc    = THEME["text_primary"] if roc >= 0.92 else THEME["danger"]
        recC  = THEME["text_primary"] if rec >= 0.82 else THEME["danger"]
        rows_html += f"""
        <tr>
            <td>{window}</td>
            <td style="color:{rc};font-weight:600;">{roc:.3f}</td>
            <td style="color:{recC};font-weight:600;">{rec:.3f}</td>
            <td>{prec:.3f}</td>
            <td>{fr}</td><td>{dr}</td>
            <td><span class="badge {badge}">{status}</span></td>
        </tr>"""

    st.markdown(f"""
    <div style="background:{THEME['bg_card']};border-radius:{THEME['radius']};
                padding:1.25rem;box-shadow:0 1px 3px rgba(0,0,0,0.06);
                border:1px solid {THEME['border']};margin-bottom:1.5rem;
                overflow-x:auto;">
        <table class="monitoring-table">
            <thead><tr>
                <th>Window</th><th>ROC-AUC</th><th>Recall</th>
                <th>Precision</th><th>Fraud Rate</th>
                <th>Detection Rate</th><th>Status</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns(2, gap="medium")
    windows_labels = ["Reference", "Window 1", "Window 2"]

    with col_left:
        st.markdown('<div class="section-title">ROC-AUC Degradation</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_hrect(y0=0.88, y1=1.0, fillcolor="rgba(79,138,239,0.05)",
                      line_width=0, annotation_text="Safe zone",
                      annotation_font_size=10,
                      annotation_font_color=THEME["text_muted"])
        fig.add_trace(go.Scatter(
            x=windows_labels, y=[0.982, 0.923, 0.910],
            mode="lines+markers",
            line=dict(color=THEME["accent"], width=2.5),
            marker=dict(size=8, color=THEME["accent"],
                        line=dict(color="white", width=2)),
            fill="tozeroy", fillcolor="rgba(79,138,239,0.06)"
        ))
        fig.add_hline(y=0.88, line_dash="dot", line_color=THEME["danger"],
                      line_width=1.5, annotation_text="Retraining trigger",
                      annotation_font_size=10,
                      annotation_font_color=THEME["danger"])
        fig.update_layout(**base_layout(height=220))
        styled_axes(fig, yrange=[0.85, 1.0])
        fig.update_yaxes(title="ROC-AUC")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title">Recall Degradation</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=windows_labels, y=[0.945, 0.817, 0.800],
            mode="lines+markers",
            line=dict(color=THEME["danger"], width=2.5),
            marker=dict(size=8, color=THEME["danger"],
                        line=dict(color="white", width=2)),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.06)"
        ))
        fig.add_hline(y=0.80, line_dash="dot", line_color=THEME["warning"],
                      line_width=1.5, annotation_text="Retraining trigger",
                      annotation_font_size=10,
                      annotation_font_color=THEME["warning"])
        fig.update_layout(**base_layout(height=220))
        styled_axes(fig, yrange=[0.75, 1.0])
        fig.update_yaxes(title="Recall")
        st.plotly_chart(fig, use_container_width=True)

    profit_img = ASSETS / "10_profit_curve.png"
    if profit_img.exists():
        st.markdown('<div class="section-title">Profit Curve — Threshold Optimization</div>',
                    unsafe_allow_html=True)
        st.image(str(profit_img), use_container_width=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Evidently AI Drift Reports</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="small")
    for col, fname, label in [
        (col1, "drift_report_window1.html", "Window 1 — Month 1"),
        (col2, "drift_report_window2.html", "Window 2 — Month 2"),
    ]:
        fpath = ASSETS / "reports" / fname
        if fpath.exists():
            with col:
                with open(fpath, "r") as f:
                    st.download_button(
                        f"Download {label}", f.read(),
                        file_name=fname, mime="text/html",
                        use_container_width=True
                    )