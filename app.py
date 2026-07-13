"""
app.py — Employee Turnover Prediction Dashboard
LozanoLsa · Project 13 · Self-Training (Semi-Supervised) · 2026 · FREE PROJECT

Algorithm: Self-Training · Base learner: Random Forest (200 trees)
Domain: HR Analytics — Turnover Risk with Unlabeled Data
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    recall_score, precision_score, confusion_matrix, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Self-Training · HR Turnover Predictor",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── FULL CSS INJECTION ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=Instrument+Serif:ital@0;1&display=swap');

:root {
    --bg:       #080c12;
    --surface:  #0e1420;
    --card:     #121922;
    --card2:    #161f2e;
    --border:   #1e2d45;
    --accent:   #a78bfa;
    --accent2:  #c4b5fd;
    --danger:   #f87171;
    --warn:     #fbbf24;
    --ok:       #4ade80;
    --text:     #c8d8f0;
    --muted:    #4e6a8a;
    --fh: 'Syne', sans-serif;
    --fm: 'JetBrains Mono', monospace;
    --fs: 'Instrument Serif', Georgia, serif;
}

.stApp { background: var(--bg) !important; color: var(--text); font-family: var(--fh); }
.block-container { padding: 1.8rem 2.4rem 3rem !important; max-width: 1400px !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1rem !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--muted) !important; letter-spacing: 0.05em; }
[data-testid="stSidebar"] label { font-family: var(--fm) !important; font-size: 0.7rem !important; color: var(--text) !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }

[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; border: 2px solid var(--accent2) !important; box-shadow: 0 0 8px rgba(167,139,250,0.5) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { font-family: var(--fm) !important; font-size: 0.65rem !important; color: var(--accent2) !important; background: var(--card) !important; border: 1px solid var(--border) !important; padding: 1px 5px !important; border-radius: 3px !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }

[data-testid="stSelectbox"] > div > div { background: var(--card) !important; border: 1px solid var(--border) !important; color: var(--text) !important; font-family: var(--fm) !important; font-size: 0.78rem !important; border-radius: 3px !important; }

[data-testid="stMetric"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-top: 2px solid var(--accent) !important; padding: 1rem 1.1rem 0.9rem !important; border-radius: 3px !important; }
[data-testid="stMetricLabel"] > div { font-family: var(--fm) !important; font-size: 0.6rem !important; text-transform: uppercase !important; letter-spacing: 0.18em !important; color: var(--muted) !important; font-weight: 400 !important; }
[data-testid="stMetricValue"] > div { font-family: var(--fm) !important; font-size: 1.7rem !important; font-weight: 600 !important; color: var(--accent2) !important; line-height: 1.1 !important; }

[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--border) !important; gap: 0 !important; background: transparent !important; }
[data-testid="stTabs"] [role="tab"] { font-family: var(--fm) !important; font-size: 0.68rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; color: var(--muted) !important; padding: 0.5rem 1.2rem !important; border: none !important; border-radius: 0 !important; background: transparent !important; transition: all 0.2s !important; }
[data-testid="stTabs"] [role="tab"]:hover { color: var(--accent2) !important; background: rgba(167,139,250,0.06) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; background: transparent !important; }
[data-testid="stTabsContent"] { padding-top: 1.4rem !important; }

[data-testid="stAlert"] { border-radius: 2px !important; font-family: var(--fm) !important; font-size: 0.75rem !important; border: none !important; }
[data-testid="stExpander"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 2px !important; }
[data-testid="stExpander"] summary { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 2px !important; }
[data-testid="stDataFrame"] th { font-family: var(--fm) !important; font-size: 0.62rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; background: var(--card2) !important; color: var(--muted) !important; }
[data-testid="stDataFrame"] td { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; background: var(--card) !important; }

hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
[data-testid="stCaptionContainer"] p { font-family: var(--fm) !important; font-size: 0.62rem !important; color: var(--muted) !important; letter-spacing: 0.08em !important; }
h1, h2, h3 { font-family: var(--fh) !important; color: var(--text) !important; }
p, li { font-family: var(--fh) !important; font-size: 0.88rem !important; }

.lsa-header { border-bottom: 1px solid var(--border); padding-bottom: 1.2rem; margin-bottom: 0.2rem; }
.lsa-project-tag { font-family: var(--fm); font-size: 0.6rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.22em; margin-bottom: 4px; }
.lsa-title { font-family: var(--fh); font-size: 1.85rem; font-weight: 800; color: #fff; line-height: 1.1; letter-spacing: -0.02em; }
.lsa-tagline { font-family: var(--fs); font-style: italic; font-size: 0.9rem; color: var(--muted); margin-top: 4px; }
.lsa-chip { display: inline-block; background: rgba(167,139,250,0.1); border: 1px solid rgba(167,139,250,0.3); color: var(--accent2); font-family: var(--fm); font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px; border-radius: 2px; margin-right: 5px; }
.lsa-chip-free { display: inline-block; background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.3); color: #4ade80; font-family: var(--fm); font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px; border-radius: 2px; margin-right: 5px; }
.lsa-section { font-family: var(--fm); font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid var(--border); }
.lsa-footer { margin-top: 2.5rem; padding-top: 0.8rem; border-top: 1px solid var(--border); font-family: var(--fm); font-size: 0.58rem; color: var(--muted); letter-spacing: 0.1em; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_PATH     = "hr_data.csv"
DATA_PATH_ALT = "13_SelfTraining_SEMI/hr_data.csv"
RANDOM_STATE  = 42
FEATURES      = ["age", "years_at_company", "job_level", "monthly_salary",
                 "job_satisfaction", "workload", "avg_overtime_hours"]
TARGET        = "resigned"
THR_HIGH      = 0.90
THR_LOW       = 0.10

FEAT_LABELS = {
    "age":                "Age",
    "years_at_company":   "Years at Company",
    "job_level":          "Job Level",
    "monthly_salary":     "Monthly Salary (MXN)",
    "job_satisfaction":   "Job Satisfaction",
    "workload":           "Workload Intensity",
    "avg_overtime_hours": "Avg Overtime (hrs/mo)",
}

# ─── MATPLOTLIB PALETTE ───────────────────────────────────────────────────────
C_BG    = "#080c12"
C_CARD  = "#121922"
C_PURP  = "#a78bfa"
C_PURP2 = "#c4b5fd"
C_DANGER= "#f87171"
C_WARN  = "#fbbf24"
C_OK    = "#4ade80"
C_TEXT  = "#c8d8f0"
C_MUTED = "#4e6a8a"
C_GRAY  = "#1e2d45"

def dark_fig(w=9, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_CARD)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.xaxis.label.set_color(C_MUTED)
    ax.yaxis.label.set_color(C_MUTED)
    ax.title.set_color(C_TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e2d45")
    return fig, ax

# ─── DATA & MODEL ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    for path in [DATA_PATH, DATA_PATH_ALT]:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue
    st.error("hr_data.csv not found. Place the file in the same folder as app.py and restart.")
    st.stop()

@st.cache_resource
def train_model(df):
    df_l = df[df[TARGET].notna()].copy()
    df_u = df[df[TARGET].isna()].copy()

    X_l = df_l[FEATURES]; y_l = df_l[TARGET].astype(int)
    X_u = df_u[FEATURES]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_l, y_l, test_size=0.2, random_state=RANDOM_STATE, stratify=y_l
    )

    # Step 1 — base model on labeled data
    base = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    base.fit(X_tr, y_tr)

    # Step 2 — pseudo-label high-confidence unlabeled
    prob_u = base.predict_proba(X_u)[:, 1]
    df_u   = df_u.copy(); df_u["p"] = prob_u
    df_ps  = df_u[(prob_u >= THR_HIGH) | (prob_u <= THR_LOW)].copy()
    df_ps[TARGET] = np.where(df_ps["p"] >= THR_HIGH, 1, 0)

    # Step 3 — final model on expanded labeled set
    X_exp = pd.concat([X_tr, df_ps[FEATURES]], ignore_index=True)
    y_exp = pd.concat([y_tr, df_ps[TARGET].astype(int)], ignore_index=True)

    final = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    final.fit(X_exp, y_exp)

    y_pred = final.predict(X_te); y_prob = final.predict_proba(X_te)[:, 1]
    metrics = {
        "Accuracy" : round(accuracy_score(y_te, y_pred), 4),
        "AUC-ROC"  : round(roc_auc_score(y_te, y_prob), 4),
        "F1 Score" : round(f1_score(y_te, y_pred), 4),
        "Precision": round(precision_score(y_te, y_pred), 4),
        "Recall"   : round(recall_score(y_te, y_pred), 4),
    }
    cm_arr = confusion_matrix(y_te, y_pred)
    imp    = (pd.DataFrame({"feature": FEATURES, "importance": final.feature_importances_})
               .sort_values("importance", ascending=False).reset_index(drop=True))
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    pseudo_info = {
        "n_pseudo": len(df_ps),
        "n_pseudo_1": int((df_ps[TARGET] == 1).sum()),
        "n_pseudo_0": int((df_ps[TARGET] == 0).sum()),
        "prob_u": prob_u,
        "expanded": len(X_exp),
    }
    return final, metrics, cm_arr, imp, fpr, tpr, X_te, y_te, y_prob, pseudo_info

df = load_data()
final_model, metrics, cm_arr, imp, fpr, tpr, X_te, y_te, y_prob, pi = train_model(df)
df_l = df[df[TARGET].notna()].copy()
df_u = df[df[TARGET].isna()].copy()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="lsa-project-tag">LozanoLsa · Project 13</div>
    <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;
                color:#fff;margin-bottom:6px;">HR Turnover<br>Predictor</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                color:#4e6a8a;line-height:1.7;margin-bottom:12px;">
        Self-Training · Random Forest<br>Semi-supervised · Free project
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="lsa-section">// Model info</div>', unsafe_allow_html=True)
    st.caption(f"Algorithm: Self-Training (Semi-Supervised)")
    st.caption(f"Base learner: Random Forest · 200 trees")
    st.caption(f"Thresholds: {THR_HIGH} / {THR_LOW}")
    st.caption(f"Labeled records: {df[TARGET].notna().sum():,}")
    st.caption(f"Unlabeled records: {df[TARGET].isna().sum():,}")
    st.caption(f"Pseudo-labels added: {pi['n_pseudo']}")
    st.caption(f"Expanded training set: {pi['expanded']:,}")
    st.divider()
    st.caption("Where f(x) meets Kaizen · 2026")

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #13 · Self-Training · Semi-Supervised · HR Analytics</div>
    <div class="lsa-title">4,000 Employees Without Labels — Not Without Signal</div>
    <div class="lsa-tagline">The model teaches itself from unlabeled data. High-confidence predictions become training examples.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">SELF-TRAINING</span>
        <span class="lsa-chip">RANDOM FOREST</span>
        <span class="lsa-chip">AUC {metrics['AUC-ROC']:.4f}</span>
        <span class="lsa-chip">{pi['n_pseudo']} PSEUDO-LABELS</span>
        <span class="lsa-chip-free">FREE PROJECT</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TOP KPI ROW ──────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Accuracy",  f"{metrics['Accuracy']:.4f}",  "Test set")
k2.metric("AUC-ROC",   f"{metrics['AUC-ROC']:.4f}",   "Ranking quality")
k3.metric("F1 Score",  f"{metrics['F1 Score']:.4f}",  "Imbalanced target")
k4.metric("Recall",    f"{metrics['Recall']:.4f}",    "Resignations caught")
k5.metric("Precision", f"{metrics['Precision']:.4f}", "Flag accuracy")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DATA EXPLORER", "PERFORMANCE", "RISK SIMULATOR", "RISK DRIVERS", "ACTION PLAN"
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="lsa-section">// Dataset overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees",   f"{len(df):,}")
    c2.metric("Labeled Records",   f"{len(df_l):,}")
    c3.metric("Unlabeled Records", f"{len(df_u):,}")
    c4.metric("Labeling Rate",     f"{len(df_l)/len(df)*100:.1f}%")

    st.divider()
    st.markdown('<div class="lsa-section">// Labeled vs unlabeled — feature distribution comparison</div>',
                unsafe_allow_html=True)
    st.caption("If both distributions overlap, pseudo-labels will generalise well to the full dataset.")

    feats_plot = ["age", "years_at_company", "job_satisfaction", "workload", "avg_overtime_hours"]
    fig, axes  = plt.subplots(1, 5, figsize=(14, 3))
    fig.patch.set_facecolor(C_BG)
    for ax, feat in zip(axes, feats_plot):
        ax.set_facecolor(C_CARD)
        ax.hist(df_l[feat], bins=15, alpha=0.75, color=C_PURP,   label="Labeled",   density=True)
        ax.hist(df_u[feat], bins=15, alpha=0.50, color=C_MUTED,  label="Unlabeled", density=True)
        ax.set_title(FEAT_LABELS.get(feat, feat.replace("_", " ")), color=C_TEXT, fontsize=8)
        ax.tick_params(colors=C_MUTED, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2d45")
        if ax == axes[0]:
            ax.legend(fontsize=7, labelcolor=C_TEXT, facecolor=C_CARD)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// Resignation rate by satisfaction & workload — labeled set</div>',
                unsafe_allow_html=True)
    st.caption("Empirical resignation rate for each combination in the labeled data.")

    pivot = (df_l.groupby(["job_satisfaction", "workload"])["resigned"]
               .mean().reset_index()
               .pivot(index="job_satisfaction", columns="workload", values="resigned"))
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    fig2.patch.set_facecolor(C_BG); ax2.set_facecolor(C_BG)
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="RdYlGn_r",
                ax=ax2, vmin=0, vmax=1, linewidths=0.5,
                cbar_kws={"label": "Resignation rate"})
    ax2.set_xlabel("Workload", color=C_MUTED)
    ax2.set_ylabel("Job Satisfaction", color=C_MUTED)
    ax2.tick_params(colors=C_MUTED)
    st.pyplot(fig2, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// Raw data sample</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True)

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="lsa-section">// Model performance — held-out test set</div>',
                unsafe_allow_html=True)
    st.caption("All metrics computed on the labeled records held out before self-training began.")

    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.markdown('<div class="lsa-section">// Confusion matrix</div>', unsafe_allow_html=True)
        tn, fp, fn, tp = cm_arr.ravel()
        fig3, ax3 = plt.subplots(figsize=(4.5, 3.5))
        fig3.patch.set_facecolor(C_BG); ax3.set_facecolor(C_BG)
        sns.heatmap(cm_arr, annot=True, fmt="d",
                    cmap=sns.light_palette("#a78bfa", as_cmap=True),
                    xticklabels=["Predicted Stay", "Predicted Resign"],
                    yticklabels=["Actual Stay", "Actual Resign"],
                    ax=ax3, linewidths=0.5, linecolor=C_BG)
        ax3.tick_params(colors=C_MUTED)
        st.pyplot(fig3, use_container_width=True); plt.close()
        st.caption(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    with col_roc:
        st.markdown('<div class="lsa-section">// ROC curve</div>', unsafe_allow_html=True)
        fig4, ax4 = dark_fig(5, 4)
        ax4.plot(fpr, tpr, color=C_PURP, linewidth=2.5,
                 label=f"AUC = {metrics['AUC-ROC']:.4f}")
        ax4.plot([0, 1], [0, 1], "--", color=C_MUTED, lw=1.2)
        ax4.fill_between(fpr, tpr, alpha=0.10, color=C_PURP)
        ax4.set_xlabel("False Positive Rate"); ax4.set_ylabel("True Positive Rate")
        ax4.legend(labelcolor=C_TEXT, facecolor=C_CARD, fontsize=8)
        fig4.tight_layout()
        st.pyplot(fig4, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// Pseudo-label confidence distribution</div>',
                unsafe_allow_html=True)
    st.caption(f"Of {len(df_u):,} unlabeled employees, {pi['n_pseudo']} fell outside the uncertain zone "
               f"({pi['n_pseudo_1']} labeled as resign, {pi['n_pseudo_0']} as stay).")

    fig5, ax5 = dark_fig(10, 3.5)
    ax5.hist(pi["prob_u"], bins=40, color=C_PURP, edgecolor="white", alpha=0.70)
    ax5.axvline(THR_HIGH, color=C_DANGER, linestyle="--", lw=2,
                label=f"High threshold: {THR_HIGH} → pseudo-label Resign")
    ax5.axvline(THR_LOW,  color=C_OK,     linestyle="--", lw=2,
                label=f"Low threshold: {THR_LOW} → pseudo-label Stay")
    ymax = ax5.get_ylim()[1] if ax5.get_ylim()[1] > 0 else 600
    ax5.fill_betweenx([0, ymax], THR_LOW, THR_HIGH,
                      alpha=0.08, color=C_WARN, label="Uncertain zone — discarded")
    ax5.set_xlabel("P(Resign) — Base Model Score")
    ax5.set_ylabel("Employee Count")
    ax5.legend(labelcolor=C_TEXT, facecolor=C_CARD, fontsize=8)
    fig5.tight_layout()
    st.pyplot(fig5, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// Metric explanations</div>', unsafe_allow_html=True)
    captions = {
        "Accuracy" : "Correct predictions out of all test records.",
        "AUC-ROC"  : "Model's ability to rank risks. 1.0 = perfect · 0.5 = random.",
        "F1 Score" : "Harmonic mean of Precision & Recall — best for imbalanced targets.",
        "Precision": "Of those flagged as resigning, how many actually resigned.",
        "Recall"   : "Of those who actually resigned, how many were caught.",
    }
    for name, expl in captions.items():
        with st.expander(f"{name}  —  {metrics[name]:.4f}"):
            st.write(expl)

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown('<div class="lsa-section">// Employee profile</div>', unsafe_allow_html=True)
        age            = st.slider("Age", 20, 60, 30)
        years_at_co    = st.slider("Years at Company", 0, 20, 4)
        job_level      = st.selectbox("Job Level", options=[1, 2, 3, 4],
                                      format_func=lambda x: {
                                          1: "1 — Operative", 2: "2 — Analyst",
                                          3: "3 — Coordinator", 4: "4 — Manager"
                                      }[x])
        monthly_salary = st.slider("Monthly Salary (MXN)", 8000, 70000, 18000, step=500)
        st.markdown('<div class="lsa-section">// Engagement & workload</div>', unsafe_allow_html=True)
        job_satisfaction = st.slider("Job Satisfaction (1=very low → 5=very high)", 1, 5, 3)
        workload         = st.slider("Workload Intensity (1=light → 5=overwhelming)", 1, 5, 3)
        avg_overtime     = st.slider("Avg Overtime Hours / Month", 0, 20, 8)

    profile = pd.DataFrame([{
        "age": age, "years_at_company": years_at_co, "job_level": job_level,
        "monthly_salary": monthly_salary, "job_satisfaction": job_satisfaction,
        "workload": workload, "avg_overtime_hours": avg_overtime,
    }])
    p_resign = final_model.predict_proba(profile)[0, 1]

    if p_resign >= 0.80:
        risk_label = "CRITICAL"; risk_color = C_DANGER; risk_bg = "#2e0f0f"
    elif p_resign >= 0.55:
        risk_label = "HIGH";     risk_color = C_WARN;   risk_bg = "#2e2a0a"
    elif p_resign >= 0.35:
        risk_label = "MODERATE"; risk_color = "#d4c34a"; risk_bg = "#2a2d10"
    else:
        risk_label = "LOW";      risk_color = C_OK;     risk_bg = "#0f2e1a"

    with col_result:
        st.markdown(
            f'''<div style="background:var(--card);border:1px solid var(--border);
                        border-radius:4px;padding:1.6rem 1.8rem;">
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                            color:#fff;margin-bottom:1rem;">Risk Assessment</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:3.4rem;
                            font-weight:700;color:{risk_color};line-height:1;
                            letter-spacing:-0.02em;">{p_resign:.1%}</div>
                <div style="margin-top:14px;">
                    <span style="background:{risk_bg};color:{risk_color};
                                 font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                 font-weight:600;letter-spacing:.1em;
                                 padding:5px 16px;border-radius:20px;">{risk_label}</span>
                </div>
                <div style="margin-top:18px;font-family:'JetBrains Mono',monospace;
                            font-size:0.68rem;color:var(--muted);line-height:2.1;">
                    Critical &#8805; 0.80 · High 0.55–0.80<br>
                    Moderate 0.35–0.55 · Low &lt; 0.35
                </div>
            </div>''',
            unsafe_allow_html=True
        )

        st.divider()
        # P(Resign) bar
        fig_g, ax_g = plt.subplots(figsize=(5, 0.9))
        fig_g.patch.set_facecolor(C_BG); ax_g.set_facecolor(C_BG)
        ax_g.barh([0], [1],        color="#1e2d45", height=0.55)
        ax_g.barh([0], [p_resign], color=risk_color, height=0.55, alpha=0.85)
        ax_g.axvline(0.35, color=C_MUTED, lw=0.8, ls=":")
        ax_g.axvline(0.55, color=C_MUTED, lw=0.8, ls=":")
        ax_g.axvline(0.80, color=C_MUTED, lw=0.8, ls=":")
        ax_g.set_xlim(0, 1); ax_g.axis("off")
        fig_g.tight_layout(pad=0)
        st.pyplot(fig_g, use_container_width=True); plt.close()

        st.markdown('<div class="lsa-section">// Top 5 risk drivers for this profile</div>',
                    unsafe_allow_html=True)
        feat_vals  = [age, years_at_co, job_level, monthly_salary,
                      job_satisfaction, workload, avg_overtime]
        driver_data = sorted(
            [{"Feature": FEAT_LABELS.get(f, f), "Value": v, "Importance": i}
             for f, v, i in zip(FEATURES, feat_vals, final_model.feature_importances_)],
            key=lambda x: x["Importance"], reverse=True
        )[:5]
        for d in driver_data:
            bar_w = int(d["Importance"] / final_model.feature_importances_.max() * 100)
            st.markdown(f"""
            <div style='margin:4px 0; display:flex; align-items:center; gap:10px;'>
                <div style='width:150px; font-family:var(--fm); font-size:0.7rem;
                            color:var(--text);'>{d["Feature"]}</div>
                <div style='flex:1; background:#1e2d45; border-radius:3px; height:8px;'>
                    <div style='width:{bar_w}%; background:var(--accent);
                                border-radius:3px; height:8px;'></div>
                </div>
                <div style='font-family:var(--fm); font-size:0.7rem;
                            color:var(--accent2); font-weight:600;'>{d["Value"]}</div>
            </div>""", unsafe_allow_html=True)

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    col_table, col_chart = st.columns([1, 1.5])

    with col_table:
        st.markdown('<div class="lsa-section">// Driver ranking — mean decrease impurity</div>',
                    unsafe_allow_html=True)
        for _, row in imp.iterrows():
            pct = row["importance"] / imp["importance"].max()
            bar = int(pct * 180)
            st.markdown(f"""
            <div style='margin:6px 0; padding:8px 12px; background:var(--card);
                        border-radius:3px; border-left:3px solid var(--accent);'>
                <div style='font-family:var(--fm); font-size:0.72rem; font-weight:600;
                            color:var(--text);'>{FEAT_LABELS.get(row["feature"], row["feature"])}</div>
                <div style='background:#1e2d45; border-radius:2px; height:6px; margin:5px 0;'>
                    <div style='width:{bar}px; max-width:100%; background:var(--accent);
                                border-radius:2px; height:6px;'></div>
                </div>
                <div style='font-family:var(--fm); font-size:0.68rem; color:var(--accent2);'>
                    {row["importance"]:.4f}</div>
            </div>""", unsafe_allow_html=True)

    with col_chart:
        st.markdown('<div class="lsa-section">// Response surface — satisfaction × workload</div>',
                    unsafe_allow_html=True)
        st.caption("P(Resign) for a reference employee (age=32, 5 yrs, level 2, salary 18k, 10 OT hrs) as satisfaction and workload vary.")
        grid_data = []
        for s in range(1, 6):
            for c in range(1, 6):
                p = final_model.predict_proba(pd.DataFrame([{
                    "age": 32, "years_at_company": 5, "job_level": 2,
                    "monthly_salary": 18000, "job_satisfaction": s,
                    "workload": c, "avg_overtime_hours": 10,
                }]))[0, 1]
                grid_data.append({"job_satisfaction": s, "workload": c, "p_resign": p})
        df_gr = pd.DataFrame(grid_data).pivot(index="job_satisfaction", columns="workload", values="p_resign")

        fig6, ax6 = plt.subplots(figsize=(5.5, 4.5))
        fig6.patch.set_facecolor(C_BG); ax6.set_facecolor(C_BG)
        sns.heatmap(df_gr, annot=True, fmt=".2f", cmap="RdYlGn_r",
                    ax=ax6, vmin=0, vmax=1,
                    cbar_kws={"label": "P(Resign)"})
        ax6.set_xlabel("Workload", color=C_MUTED)
        ax6.set_ylabel("Job Satisfaction", color=C_MUTED)
        ax6.tick_params(colors=C_MUTED)
        ax6.set_title("P(Resign): Satisfaction × Workload", color=C_TEXT, fontsize=10)
        st.pyplot(fig6, use_container_width=True); plt.close()

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="lsa-section">// HR action plan by risk tier</div>',
                unsafe_allow_html=True)
    actions = [
        (C_DANGER,  "CRITICAL (P ≥ 0.80)",
         "Schedule a structured 1:1 within 72 hours. Address job satisfaction and workload directly. "
         "Prepare retention offer if applicable. Losing this employee now carries the highest replacement cost."),
        (C_WARN,    "HIGH (0.55 ≤ P < 0.80)",
         "Review workload distribution this sprint cycle. Confirm career path conversation has occurred "
         "in the last 90 days. Flag for People Partner follow-up. Do not wait for the next annual review."),
        ("#d4c34a", "MODERATE (0.35 ≤ P < 0.55)",
         "Include in next pulse survey cohort. Monitor satisfaction trend. Ensure overtime hours have not "
         "been persistently above 10/month. This group is often the most salvageable with low-cost interventions."),
        (C_OK,      "LOW (P < 0.35)",
         "Maintain regular engagement cadence. Re-score monthly as workload or team changes occur. "
         "These employees can shift risk tiers quickly during organisational change — keep the model refreshed."),
        (C_PURP,    "Unlabeled (no resignation record)",
         "The unlabeled employees are not passive — they are unobserved. Run a structured satisfaction survey "
         "to convert them to labeled data. Each label improves the model's ability to serve the entire organisation."),
    ]
    for color, tier, text in actions:
        st.markdown(f"""
        <div style='margin:8px 0; padding:1.1rem 1.3rem; background:var(--card);
                    border-radius:2px; border-left:3px solid {color};'>
            <div style='font-family:var(--fm); font-size:0.72rem; font-weight:600;
                        color:{color}; margin-bottom:6px;'>{tier}</div>
            <div style='font-family:var(--fm); font-size:0.7rem; color:var(--muted);
                        line-height:1.7;'>{text}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);
                border-left:3px solid {C_PURP};border-radius:2px;
                padding:1rem 1.3rem;">
        <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Ethical use reminder</div>
        <div style="font-family:var(--fm);font-size:0.72rem;color:var(--text);line-height:1.7;">
            This model surfaces statistical patterns — not individual judgments.
            Never use a turnover probability score to deny opportunities, reduce
            investment in an employee, or pre-emptively terminate a position.
            The model's job is to trigger a conversation, not to replace one.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="background:var(--card);border:1px solid var(--border);border-radius:2px;
                padding:1rem 1.3rem;text-align:center;">
        <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Free project</div>
        <div style="font-family:var(--fm);font-size:0.68rem;color:var(--muted);line-height:1.7;">
            Full dataset + simulator included. Check the rest of the portfolio at
            <span style="color:#a78bfa;">lozanolsa.gumroad.com</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · HR Turnover Predictor · Project 13 · v2.0
</div>
""", unsafe_allow_html=True)
