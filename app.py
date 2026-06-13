from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import hashlib
import json
import platform

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import sklearn
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.pipeline import Pipeline

from ml_pipeline import (
    PRIMARY_METRIC,
    RANDOM_STATE,
    build_optuna_estimator,
    fit_default_model,
    get_models,
    get_preprocessor,
    load_data,
    prepare_features,
    split_columns,
    train_models,
)

st.set_page_config(
    page_title="Student Performance ML",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --navy: #0f172a;
            --blue: #2563eb;
            --teal: #0f766e;
            --sky: #e0f2fe;
            --slate: #475569;
            --muted: #64748b;
            --border: #e2e8f0;
            --surface: #ffffff;
            --soft: #f8fafc;
            --danger: #b91c1c;
            --warning: #b45309;
            --success: #047857;
        }
        .block-container {
            max-width: 1460px;
            padding-top: 1.1rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #334155;
        }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,.92);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            box-shadow: 0 6px 18px rgba(15,23,42,.05);
        }
        [data-testid="stMetricLabel"] {color: var(--muted); font-weight: 650;}
        [data-testid="stMetricValue"] {font-size: 1.65rem; color: var(--navy);}
        div.stButton > button, div.stDownloadButton > button {
            border-radius: 12px;
            min-height: 2.8rem;
            font-weight: 700;
            border: 1px solid #cbd5e1;
        }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #2563eb, #0f766e);
            border: 0;
            box-shadow: 0 8px 20px rgba(37,99,235,.2);
        }
        div[data-baseweb="tab-list"] {gap: .45rem;}
        button[data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            font-weight: 650;
        }
        .hero {
            background:
                radial-gradient(circle at 90% 10%, rgba(56,189,248,.30), transparent 34%),
                linear-gradient(120deg, #0f172a 0%, #172554 52%, #0f766e 100%);
            color: white;
            border-radius: 24px;
            padding: 2.1rem 2.2rem;
            margin: .25rem 0 1.25rem 0;
            box-shadow: 0 18px 42px rgba(15,23,42,.18);
        }
        .hero h1 {font-size: 2.2rem; margin: .25rem 0 .55rem 0; line-height: 1.12;}
        .hero p {max-width: 920px; color: #dbeafe; font-size: 1.03rem; margin: 0;}
        .eyebrow {
            display: inline-block;
            letter-spacing: .12em;
            text-transform: uppercase;
            font-size: .74rem;
            font-weight: 800;
            color: #bae6fd;
        }
        .badge-row {display:flex; gap:.55rem; flex-wrap:wrap; margin-top:1.1rem;}
        .badge {
            background: rgba(255,255,255,.12);
            border: 1px solid rgba(255,255,255,.22);
            padding: .38rem .68rem;
            border-radius: 999px;
            font-size: .78rem;
            color: #f8fafc;
        }
        .panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 6px 18px rgba(15,23,42,.045);
            height: 100%;
        }
        .panel h3 {margin-top:0; color:var(--navy);}
        .callout {
            border-left: 5px solid var(--blue);
            background: #eff6ff;
            border-radius: 10px;
            padding: .9rem 1rem;
            color: #1e3a8a;
            margin: .75rem 0;
        }
        .callout-risk {border-left-color: var(--warning); background:#fff7ed; color:#7c2d12;}
        .callout-success {border-left-color: var(--success); background:#ecfdf5; color:#064e3b;}
        .section-kicker {color:var(--blue); font-weight:800; font-size:.78rem; letter-spacing:.09em; text-transform:uppercase;}
        .section-subtitle {color:var(--muted); margin-top:-.35rem; margin-bottom:1rem;}
        .small-note {font-size: .88rem; color: var(--muted);}
        .mono-card {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: .82rem;
            background:#0f172a;
            color:#dbeafe;
            border-radius:14px;
            padding:1rem;
            overflow:auto;
        }
        .journey {display:grid; grid-template-columns: repeat(5, 1fr); gap:.7rem; margin:1rem 0;}
        .journey-step {background:#ffffff; border:1px solid #dbeafe; border-radius:16px; padding:1rem; box-shadow:0 8px 18px rgba(15,23,42,.045);}
        .journey-step b {color:#0f172a;}
        .journey-step span {display:block; color:#64748b; font-size:.84rem; margin-top:.35rem;}
        .signature-strip {background:linear-gradient(90deg,#0f172a,#1d4ed8,#0f766e); border-radius:18px; padding:1rem 1.15rem; color:white; margin:.8rem 0 1.1rem;}
        .signature-strip strong {color:#bae6fd;}
        .status-green {background:#ecfdf5; border:1px solid #a7f3d0; color:#065f46; padding:.55rem .75rem; border-radius:999px; font-weight:800; display:inline-block;}
        .status-amber {background:#fffbeb; border:1px solid #fde68a; color:#92400e; padding:.55rem .75rem; border-radius:999px; font-weight:800; display:inline-block;}
        .status-red {background:#fef2f2; border:1px solid #fecaca; color:#991b1b; padding:.55rem .75rem; border-radius:999px; font-weight:800; display:inline-block;}

        .top-status-bar {
            position: sticky;
            top: 0;
            z-index: 999;
            background: rgba(248,250,252,.86);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(226,232,240,.85);
            border-radius: 18px;
            padding: .72rem .95rem;
            margin: 0 0 1rem 0;
            box-shadow: 0 8px 26px rgba(15,23,42,.06);
        }
        .top-status-grid {display:grid; grid-template-columns: 1.1fr repeat(4, .7fr); gap:.65rem; align-items:center;}
        .status-chip {
            border-radius: 999px;
            padding: .44rem .62rem;
            background: #ffffff;
            border: 1px solid #dbeafe;
            color: #1e293b;
            font-size: .78rem;
            font-weight: 800;
            text-align:center;
            white-space: nowrap;
        }
        .status-chip strong {color:#2563eb;}
        .nav-card {
            background: rgba(255,255,255,.72);
            border: 1px solid #dbeafe;
            border-radius: 18px;
            padding: .75rem .7rem;
            margin-bottom: .85rem;
            box-shadow: 0 10px 28px rgba(15,23,42,.055);
        }
        .nav-title {
            color:#0f172a;
            font-weight: 900;
            font-size:.82rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin: .35rem 0 .45rem;
        }
        .nav-active {
            background: linear-gradient(90deg,#2563eb,#0f766e);
            color:white;
            border-radius:12px;
            padding:.58rem .68rem;
            font-weight:900;
            margin:.25rem 0 .4rem;
            box-shadow: 0 8px 20px rgba(37,99,235,.18);
        }
        .executive-card {
            background: linear-gradient(180deg,#ffffff 0%,#f8fafc 100%);
            border: 1px solid #dbeafe;
            border-radius: 22px;
            padding: 1.15rem;
            box-shadow: 0 12px 30px rgba(15,23,42,.06);
            height:100%;
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }
        .executive-card:hover {
            transform: translateY(-3px);
            border-color:#93c5fd;
            box-shadow: 0 20px 42px rgba(15,23,42,.10);
        }
        .executive-card h4 {margin:.1rem 0 .35rem; color:#0f172a;}
        .executive-card p {color:#475569; font-size:.92rem; margin:.35rem 0 0;}
        .story-box {
            background: radial-gradient(circle at 100% 10%, rgba(14,165,233,.16), transparent 28%), #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 24px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 14px 36px rgba(15,23,42,.06);
            margin: .8rem 0 1rem;
        }
        .story-box h3 {margin-top:0; color:#0f172a;}
        .action-tile {
            background:#ffffff;
            border:1px solid #e2e8f0;
            border-radius:18px;
            padding:1rem;
            height:100%;
        }
        .action-tile b {color:#0f172a;}
        .action-tile span {display:block; color:#64748b; font-size:.88rem; margin-top:.25rem;}
        .pulse-dot {
            display:inline-block;
            width:.62rem;
            height:.62rem;
            border-radius:999px;
            background:#10b981;
            box-shadow:0 0 0 0 rgba(16,185,129,.55);
            animation:pulse 1.8s infinite;
            margin-right:.4rem;
        }
        @keyframes pulse {
            0% {box-shadow:0 0 0 0 rgba(16,185,129,.55);}
            70% {box-shadow:0 0 0 9px rgba(16,185,129,0);}
            100% {box-shadow:0 0 0 0 rgba(16,185,129,0);}
        }
        .scenario-card {
            border-radius: 20px;
            padding: 1rem 1.1rem;
            background: #ffffff;
            border: 1px solid #dbeafe;
            box-shadow: 0 10px 24px rgba(15,23,42,.05);
        }
        .scenario-card .delta-good {color:#047857; font-weight:900;}
        .scenario-card .delta-bad {color:#b91c1c; font-weight:900;}
        .control-panel {
            background: linear-gradient(135deg,#0f172a,#172554);
            color:#e0f2fe;
            border-radius: 24px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 18px 40px rgba(15,23,42,.18);
        }
        .control-panel h3 {color:white; margin-top:0;}
        .control-panel p {color:#bfdbfe;}
        .workflow-pill {
            display:inline-flex;
            align-items:center;
            gap:.45rem;
            border:1px solid #bfdbfe;
            background:#eff6ff;
            color:#1d4ed8;
            padding:.48rem .72rem;
            border-radius:999px;
            font-weight:800;
            font-size:.82rem;
            margin:.2rem;
        }
        @media (max-width: 1000px) {.top-status-grid {grid-template-columns:1fr 1fr;}}

        @media (max-width: 1000px) {.journey {grid-template-columns:1fr;}}
        @media (max-width: 900px) {
            .hero {padding:1.45rem; border-radius:18px;}
            .hero h1 {font-size:1.65rem;}
        }
    </style>
    """,
    unsafe_allow_html=True,
)

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    "#2563eb", "#0f766e", "#7c3aed", "#ea580c", "#dc2626", "#0891b2", "#64748b"
]


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource(show_spinner=False)
def get_cached_default_model(X_data: pd.DataFrame, y_data: np.ndarray):
    cat, num = split_columns(X_data)
    prep = get_preprocessor(cat, num)
    return fit_default_model(X_data, y_data, prep)


def metric_mean(bundle: dict, model_name: str, metric: str) -> float:
    return float(bundle["results"][model_name][metric]["mean"])


def metric_std(bundle: dict, model_name: str, metric: str) -> float:
    return float(bundle["results"][model_name][metric]["std"])


def format_result_table(bundle: dict) -> pd.DataFrame:
    rows = []
    primary_metric = bundle.get("primary_metric", PRIMARY_METRIC)
    sorted_models = sorted(
        bundle["results"],
        key=lambda name: metric_mean(bundle, name, primary_metric),
        reverse=True,
    )
    for rank, name in enumerate(sorted_models, start=1):
        result = bundle["results"][name]
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")
        rows.append(
            {
                "Rank": f"{medal} {rank}".strip(),
                "Model": name,
                "Macro-F1": f"{result['Macro-F1']['mean']:.3f} ± {result['Macro-F1']['std']:.3f}",
                "Risk Recall": f"{result['Risk Recall']['mean']:.3f} ± {result['Risk Recall']['std']:.3f}",
                "F1-Risk": f"{result['F1-Risk']['mean']:.3f} ± {result['F1-Risk']['std']:.3f}",
                "Balanced Accuracy": f"{result['Balanced Accuracy']['mean']:.3f} ± {result['Balanced Accuracy']['std']:.3f}",
                "AUC-ROC": f"{result['AUC-ROC']['mean']:.3f} ± {result['AUC-ROC']['std']:.3f}",
                "MCC": f"{result['MCC']['mean']:.3f} ± {result['MCC']['std']:.3f}",
            }
        )
    return pd.DataFrame(rows)

def arrow_safe_df(data) -> pd.DataFrame:
    """Return a Streamlit/PyArrow-safe DataFrame without mixed object columns."""
    if hasattr(data, "data") and isinstance(getattr(data, "data"), pd.DataFrame):
        frame = data.data.copy()
    else:
        frame = pd.DataFrame(data).copy()

    for column in frame.columns:
        series = frame[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            frame[column] = series.astype(str)
        elif series.dtype == "object":
            def normalize(value):
                if value is None:
                    return ""
                if isinstance(value, (dict, list, tuple, set, np.ndarray)):
                    return str(value)
                try:
                    if pd.isna(value):
                        return ""
                except (TypeError, ValueError):
                    pass
                return str(value)
            frame[column] = series.map(normalize)
    return frame


def model_to_bytes(model) -> bytes:
    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def render_page_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div style="margin-bottom:1rem">
            <div class="section-kicker">{kicker}</div>
            <h1 style="margin:.2rem 0 .35rem 0;color:#0f172a">{title}</h1>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()[:16]


def oof_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    risk_recall = tn / (tn + fp) if (tn + fp) else 0.0
    missed_risk = fp / (tn + fp) if (tn + fp) else 0.0
    false_alarm = fn / (fn + tp) if (fn + tp) else 0.0
    risk_true = 1 - y_true
    risk_prob = 1 - y_prob
    return {
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Risk Recall": float(risk_recall),
        "F1-Risk": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "Missed Risk Rate": float(missed_risk),
        "False Alarm Rate": float(false_alarm),
        "AUC-ROC": float(auc(*roc_curve(y_true, y_prob)[:2])),
        "PR-AUC-Risk": float(average_precision_score(risk_true, risk_prob)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def bootstrap_confidence_intervals(
    y_true,
    y_prob,
    threshold: float = 0.5,
    iterations: int = 500,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    estimates = oof_metrics(y_true, y_prob, threshold)
    tracked = ["Macro-F1", "Balanced Accuracy", "Risk Recall", "F1-Risk", "AUC-ROC", "MCC"]
    samples = {metric: [] for metric in tracked}
    rng = np.random.default_rng(RANDOM_STATE)

    for _ in range(iterations):
        indices = rng.integers(0, len(y_true), len(y_true))
        sampled_true = y_true[indices]
        sampled_prob = y_prob[indices]
        if len(np.unique(sampled_true)) < 2:
            continue
        values = oof_metrics(sampled_true, sampled_prob, threshold)
        for metric in tracked:
            samples[metric].append(values[metric])

    rows = []
    for metric in tracked:
        values = np.asarray(samples[metric], dtype=float)
        lower, upper = np.percentile(values, [2.5, 97.5]) if len(values) else (np.nan, np.nan)
        rows.append(
            {
                "Metric": metric,
                "Estimate": estimates[metric],
                "95% CI lower": float(lower),
                "95% CI upper": float(upper),
            }
        )
    return pd.DataFrame(rows)


def quality_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"Check": "Rows", "Value": int(len(frame)), "Status": "Info"},
        {"Check": "Columns", "Value": int(frame.shape[1]), "Status": "Info"},
        {"Check": "Missing cells", "Value": int(frame.isna().sum().sum()), "Status": "Review" if frame.isna().sum().sum() else "Pass"},
        {"Check": "Duplicate rows", "Value": int(frame.duplicated().sum()), "Status": "Review" if frame.duplicated().sum() else "Pass"},
        {"Check": "Constant columns", "Value": int((frame.nunique(dropna=False) <= 1).sum()), "Status": "Review" if (frame.nunique(dropna=False) <= 1).sum() else "Pass"},
        {"Check": "Dataset fingerprint", "Value": dataset_fingerprint(frame), "Status": "Reproducibility"},
    ]
    return pd.DataFrame(rows)


def performance_statement(bundle: dict) -> tuple[str, str]:
    best = bundle["best_model_name"]
    baseline = "Dummy Baseline"
    best_score = metric_mean(bundle, best, PRIMARY_METRIC)
    baseline_score = metric_mean(bundle, baseline, PRIMARY_METRIC)
    uplift = best_score - baseline_score
    risk_recall = metric_mean(bundle, best, "Risk Recall")

    if best_score >= 0.75 and risk_recall >= 0.70 and uplift >= 0.10:
        return (
            "Strong prototype evidence",
            f"{best} improves Macro-F1 over the baseline by {uplift:.3f} and achieves risk recall of {risk_recall:.3f}. Production readiness is not claimed without external validation.",
        )
    if best_score >= 0.60 and uplift >= 0.05:
        return (
            "Moderate predictive evidence",
            f"{best} improves Macro-F1 over the baseline by {uplift:.3f}. The result is useful for an academic prototype but is insufficient for operational university decisions.",
        )
    return (
        "Limited predictive evidence",
        f"The Macro-F1 uplift over the baseline is {uplift:.3f}. The defensible conclusion is to report the model's limitations rather than present it as a strong predictor.",
    )


def holm_adjusted_pvalues(p_values: list[float]) -> np.ndarray:
    """Return Holm-Bonferroni adjusted p-values while preserving input order."""
    raw = np.asarray(p_values, dtype=float)
    if raw.size == 0:
        return raw
    order = np.argsort(raw)
    adjusted = np.empty_like(raw)
    running_max = 0.0
    total = len(raw)
    for rank, original_index in enumerate(order):
        candidate = min((total - rank) * raw[original_index], 1.0)
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def pairwise_wilcoxon_results(bundle: dict) -> pd.DataFrame:
    """Compare all model pairs on matched outer-fold Macro-F1 scores."""
    from itertools import combinations

    rows = []
    names = list(bundle["score_arrays"].keys())
    for model_a, model_b in combinations(names, 2):
        scores_a = np.asarray(bundle["score_arrays"][model_a], dtype=float)
        scores_b = np.asarray(bundle["score_arrays"][model_b], dtype=float)
        difference = scores_a - scores_b
        if np.allclose(difference, 0):
            statistic, p_value = 0.0, 1.0
        else:
            try:
                statistic, p_value = wilcoxon(scores_a, scores_b)
            except ValueError:
                statistic, p_value = np.nan, 1.0
        rows.append(
            {
                "Model A": model_a,
                "Model B": model_b,
                "Mean difference": float(difference.mean()),
                "Wilcoxon statistic": float(statistic) if np.isfinite(statistic) else np.nan,
                "Raw p-value": float(p_value),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["Holm-adjusted p-value"] = holm_adjusted_pvalues(
            result["Raw p-value"].tolist()
        )
        result["Significant at 0.05"] = result["Holm-adjusted p-value"] < 0.05
    return result


def threshold_cost_analysis(
    y_true,
    y_prob_pass,
    missed_risk_cost: float,
    false_alarm_cost: float,
) -> pd.DataFrame:
    """Evaluate operational error cost across pass-probability thresholds."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob_pass = np.asarray(y_prob_pass, dtype=float)
    rows = []
    for threshold in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob_pass >= threshold).astype(int)
        missed_risk = int(((y_true == 0) & (y_pred == 1)).sum())
        false_alarm = int(((y_true == 1) & (y_pred == 0)).sum())
        total_cost = missed_risk * missed_risk_cost + false_alarm * false_alarm_cost
        metrics = oof_metrics(y_true, y_prob_pass, float(threshold))
        rows.append(
            {
                "Pass threshold": float(threshold),
                "Missed-risk cases": missed_risk,
                "False alarms": false_alarm,
                "Weighted cost": float(total_cost),
                "Cost per student": float(total_cost / len(y_true)),
                "Macro-F1": metrics["Macro-F1"],
                "Risk Recall": metrics["Risk Recall"],
            }
        )
    return pd.DataFrame(rows)


def build_executive_summary(bundle: dict, frame: pd.DataFrame, features: pd.DataFrame) -> str:
    best_name = bundle["best_model_name"]
    baseline = "Dummy Baseline"
    best_macro = metric_mean(bundle, best_name, "Macro-F1")
    baseline_macro = metric_mean(bundle, baseline, "Macro-F1")
    risk_recall = metric_mean(bundle, best_name, "Risk Recall")
    auc_score = metric_mean(bundle, best_name, "AUC-ROC")
    title, statement = performance_statement(bundle)
    return f"""# Executive Summary — Explainable Student Risk Analytics

## Project purpose
This Pearson BTEC Level 6 applied-AI prototype evaluates whether machine-learning models can identify students at risk of failing using the UCI Student Performance dataset. The system is designed for research and human-reviewed support, not automated academic decisions.

## Dataset and experimental design
- Records: {len(frame)}
- Model inputs: {features.shape[1]}
- Target: 0 = Fail/at risk, 1 = Pass
- Prior and final grades excluded: G1, G2 and G3
- Validation: {bundle['outer_splits']} outer folds × {bundle['inner_splits']} inner folds
- Primary selection metric: {bundle.get('primary_metric', PRIMARY_METRIC)}
- Dataset fingerprint: {dataset_fingerprint(frame)}

## Main result
- Selected model: {best_name}
- Macro-F1: {best_macro:.3f}
- Dummy-baseline Macro-F1: {baseline_macro:.3f}
- Uplift over baseline: {best_macro - baseline_macro:.3f}
- Risk recall: {risk_recall:.3f}
- ROC-AUC: {auc_score:.3f}

**Interpretation:** {title}. {statement}

## Evidence produced
The application provides nested cross-validation, baseline comparison, risk-focused error analysis, calibration, bootstrap confidence intervals, subgroup diagnostics, SHAP explanations, threshold-cost analysis and an auditable evidence pack.

## Critical limitation
The dataset is historical and does not contain PDP University students. External deployment is not justified without local validation, governance review and ongoing performance monitoring.
"""


def build_evidence_pack(bundle: dict, frame: pd.DataFrame, features: pd.DataFrame) -> bytes:
    best_name = bundle["best_model_name"]
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "leaderboard.csv",
            format_result_table(bundle).to_csv(index=False),
        )
        archive.writestr(
            "fold_metrics.csv",
            bundle["fold_metrics"].to_csv(index=False),
        )
        archive.writestr(
            "oof_predictions.csv",
            pd.DataFrame(
                {
                    "y_true": bundle["oof"][best_name]["y_true"],
                    "y_pred": bundle["oof"][best_name]["y_pred"],
                    "pass_probability": bundle["oof"][best_name]["y_prob"],
                    "risk_probability": 1 - bundle["oof"][best_name]["y_prob"],
                }
            ).to_csv(index=False),
        )
        archive.writestr(
            "best_parameters.json",
            json.dumps(bundle["results"][best_name]["best_params"], indent=2, default=str),
        )
        archive.writestr(
            "experiment_metadata.json",
            json.dumps(
                {
                    "trained_at_utc": bundle.get("trained_at_utc"),
                    "random_state": RANDOM_STATE,
                    "outer_folds": bundle["outer_splits"],
                    "inner_folds": bundle["inner_splits"],
                    "primary_metric": bundle.get("primary_metric", PRIMARY_METRIC),
                    "best_model": best_name,
                    "dataset_rows": len(frame),
                    "model_features": features.shape[1],
                    "dataset_fingerprint": dataset_fingerprint(frame),
                    "python": platform.python_version(),
                    "streamlit": st.__version__,
                    "scikit_learn": sklearn.__version__,
                    "plotly": plotly.__version__,
                },
                indent=2,
            ),
        )
        statement_title, statement_text = performance_statement(bundle)
        archive.writestr(
            "model_card.md",
            f"""# Student Performance Early-Warning Model Card

## Intended use
Academic early-warning prototype. It must not be the sole basis for academic sanctions or automatic decisions.

## Dataset
UCI Student Performance dataset; not PDP University student data. Target: 0=Fail/at-risk, 1=Pass. G1, G2 and G3 are excluded from model inputs.

## Validation
{bundle['outer_splits']} outer folds x {bundle['inner_splits']} inner folds; primary metric: {bundle.get('primary_metric', PRIMARY_METRIC)}.

## Selected model
{best_name}

## Critical interpretation
**{statement_title}.** {statement_text}

## Limitations
Small historical dataset, no causal inference, no external validation, subgroup estimates may be unstable, and model performance may not generalise to other institutions.
""",
        )
        archive.writestr(
            "executive_summary.md",
            build_executive_summary(bundle, frame, features),
        )
        archive.writestr(
            "pairwise_wilcoxon_holm.csv",
            pairwise_wilcoxon_results(bundle).to_csv(index=False),
        )
        archive.writestr("best_model.joblib", model_to_bytes(bundle["best_pipes"][best_name]))
    buffer.seek(0)
    return buffer.getvalue()


FEATURE_LABELS = {
    "school": "School",
    "sex": "Sex",
    "age": "Age",
    "address": "Home area",
    "famsize": "Family size",
    "Pstatus": "Parents' status",
    "Medu": "Mother's education",
    "Fedu": "Father's education",
    "Mjob": "Mother's job",
    "Fjob": "Father's job",
    "reason": "School choice reason",
    "guardian": "Guardian",
    "traveltime": "Travel time",
    "studytime": "Weekly study time",
    "failures": "Previous class failures",
    "schoolsup": "School support",
    "famsup": "Family support",
    "paid": "Paid classes",
    "activities": "Extra-curricular activities",
    "nursery": "Nursery attendance",
    "higher": "Plans higher education",
    "internet": "Internet access",
    "romantic": "Romantic relationship",
    "famrel": "Family relationship quality",
    "freetime": "Free time",
    "goout": "Going out",
    "Dalc": "Workday alcohol use",
    "Walc": "Weekend alcohol use",
    "health": "Health status",
    "absences": "Absences",
}


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def normalize_shap_explanation(explanation, feature_names):
    import shap

    values = np.asarray(explanation.values)
    data = np.asarray(explanation.data)
    base_values = np.asarray(explanation.base_values)

    if values.ndim == 3:
        class_index = 1 if values.shape[2] > 1 else 0
        values = values[:, :, class_index]
        if base_values.ndim == 2:
            base_values = base_values[:, class_index]
        elif base_values.ndim == 1 and len(base_values) > 1:
            base_values = np.repeat(base_values[class_index], values.shape[0])

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=data,
        feature_names=list(feature_names),
    )


try:
    df = get_data()
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.code(
        "Place student-mat.csv in the repository root or "
        "use data/student-mat.csv."
    )
    st.stop()

# Early-warning setup: target=G3>=10, while G1/G2/G3 are excluded from X.
X, y = prepare_features(df, include_prior_grades=False)
cat_cols, num_cols = split_columns(X)
preprocessor = get_preprocessor(cat_cols, num_cols)

PAGES = [
    "🏠 Overview",
    "📊 EDA & Data Audit",
    "⚡ Optuna Optimization",
    "🤖 Model Training",
    "📈 Results",
    "⚖️ Fairness Analysis",
    "🧪 Validity & Error Analysis",
    "🔬 SHAP Values",
    "🔍 Prediction",
    "🎛️ Intervention Simulator",
    "🧭 Evidence Control Room",
    "ℹ️ Model Card",
    "📚 Research Evidence",
]

NAV_GROUPS = {
    "Operate": ["🏠 Overview", "📊 EDA & Data Audit", "🤖 Model Training", "📈 Results"],
    "Trust": ["⚖️ Fairness Analysis", "🧪 Validity & Error Analysis", "🔬 SHAP Values"],
    "Use": ["🔍 Prediction", "🎛️ Intervention Simulator"],
    "Evidence": ["🧭 Evidence Control Room", "ℹ️ Model Card", "📚 Research Evidence"],
    "Optional": ["⚡ Optuna Optimization"],
}

if "active_page" not in st.session_state:
    st.session_state.active_page = "🏠 Overview"
if st.session_state.active_page not in PAGES:
    st.session_state.active_page = "🏠 Overview"

with st.sidebar:
    st.markdown("## 🧠 Academic Risk Command Center")
    st.caption("Signature dashboard · responsible AI · BTEC evidence")
    st.markdown("**PDP University · BTEC Level 6 · 2026**")
    st.markdown("**Eltezarv Doniyorbek · Group 22-305**")
    st.markdown(
        f"""<div class="nav-card">
            <div class="nav-title">Current module</div>
            <div class="nav-active">{st.session_state.active_page}</div>
            <span class="small-note">Use the command navigation below. Each click changes the working mode of the dashboard.</span>
        </div>""",F
        unsafe_allow_html=True,
    )

    for group_name, items in NAV_GROUPS.items():
        st.markdown(f'<div class="nav-title">{group_name}</div>', unsafe_allow_html=True)
        for item in items:
            btn_type = "primary" if item == st.session_state.active_page else "secondary"
            if st.button(item, key=f"nav_{item}", width="stretch", type=btn_type):
                st.session_state.active_page = item
                st.rerun()

    st.divider()
    presentation_mode = st.toggle(
        "Presentation mode",
        value=st.session_state.get("presentation_mode", False),
        help="Uses tighter executive wording and keeps examiner-facing panels visible.",
    )
    st.session_state.presentation_mode = presentation_mode
    st.caption(f"Dataset: {len(df)} rows · {X.shape[1]} model features")
    st.caption("Target: 0 = Fail/at-risk · 1 = Pass")
    st.caption("Primary metric: Macro-F1")
    st.caption("G1, G2 and G3 excluded")
    st.caption(f"Fingerprint: {dataset_fingerprint(df)}")

page = st.session_state.active_page
bundle = st.session_state.get("training_bundle")

trained_state = "Ready" if bundle else "Not trained"
best_state = bundle["best_model_name"] if bundle else "Run training"
macro_state = f"{metric_mean(bundle, best_state, 'Macro-F1'):.3f}" if bundle else "—"
risk_state = f"{metric_mean(bundle, best_state, 'Risk Recall'):.3f}" if bundle else "—"
st.markdown(
    f"""<div class="top-status-bar">
        <div class="top-status-grid">
            <div><span class="pulse-dot"></span><b>Academic Risk Command Center</b></div>
            <div class="status-chip">Experiment: <strong>{trained_state}</strong></div>
            <div class="status-chip">Best model: <strong>{best_state}</strong></div>
            <div class="status-chip">Macro-F1: <strong>{macro_state}</strong></div>
            <div class="status-chip">Risk recall: <strong>{risk_state}</strong></div>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# ════════════════════════════════════════
# 🏠 OVERVIEW
# ════════════════════════════════════════
if page == "🏠 Overview":
    best_name = bundle["best_model_name"] if bundle else "Not trained"
    best_macro = metric_mean(bundle, best_name, "Macro-F1") if bundle else None
    best_risk_recall = metric_mean(bundle, best_name, "Risk Recall") if bundle else None
    baseline_macro = (
        metric_mean(bundle, "Dummy Baseline", "Macro-F1") if bundle else None
    )
    uplift = best_macro - baseline_macro if bundle else None

    st.markdown(
        """
        <div class="hero">
            <span class="eyebrow">Pearson BTEC Level 6 · Applied AI Capstone</span>
            <h1>Academic Risk Command Center</h1>
            <p>
                A signature BTEC Level 6 decision-support prototype that does more than predict pass/fail: 
                it audits data quality, compares models against a baseline, explains risk drivers, tests fairness, 
                calibrates probabilities and simulates practical student-support interventions.
            </p>
            <div class="badge-row">
                <span class="badge">Nested cross-validation</span>
                <span class="badge">Risk-focused Macro-F1</span>
                <span class="badge">Fairness analysis</span>
                <span class="badge">SHAP explanations</span>
                <span class="badge">Responsible AI</span>
                <span class="badge">Intervention simulator</span>
                <span class="badge">Evidence control room</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    story_mode = st.radio(
        "Choose presentation angle",
        ["Executive story", "Technical proof", "Viva defence"],
        horizontal=True,
        help="This changes the front-end narrative without changing the model results.",
    )
    if story_mode == "Executive story":
        st.markdown(
            """
            <div class="story-box">
                <h3>From prediction to student support</h3>
                <p>This prototype is designed as an academic-risk operations dashboard. It begins with data audit, trains leakage-safe models, compares them with a baseline, then turns predictions into explanations, fairness checks and support scenarios.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif story_mode == "Technical proof":
        st.markdown(
            """
            <div class="story-box">
                <h3>Technical credibility layer</h3>
                <p>The app uses nested cross-validation, Macro-F1 model selection, out-of-fold predictions, threshold-cost analysis, calibration, SHAP and subgroup diagnostics. This is designed to avoid the common mistake of presenting an attractive dashboard without defensible validation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="story-box">
                <h3>Defence argument</h3>
                <p>The central claim is deliberately cautious: the model provides moderate early-warning signals, not automatic decisions. The value of the project is the responsible ML workflow and deployed evidence system, not an exaggerated accuracy claim.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    cta_cols = st.columns(4)
    with cta_cols[0]:
        if st.button("🚀 Train models", width="stretch", type="primary"):
            st.session_state.active_page = "🤖 Model Training"
            st.rerun()
    with cta_cols[1]:
        if st.button("📈 Inspect results", width="stretch"):
            st.session_state.active_page = "📈 Results"
            st.rerun()
    with cta_cols[2]:
        if st.button("🎛️ Simulate support", width="stretch"):
            st.session_state.active_page = "🎛️ Intervention Simulator"
            st.rerun()
    with cta_cols[3]:
        if st.button("🧭 Open evidence", width="stretch"):
            st.session_state.active_page = "🧭 Evidence Control Room"
            st.rerun()

    cols = st.columns(5)
    cols[0].metric("Students", len(df))
    cols[1].metric("Model features", X.shape[1])
    cols[2].metric("Candidate models", f"{len(get_models()) - 1} + baseline")
    cols[3].metric(
        "Best Macro-F1",
        f"{best_macro:.3f}" if best_macro is not None else "—",
        delta=f"+{uplift:.3f} vs baseline" if uplift is not None else None,
    )
    cols[4].metric(
        "Risk recall",
        f"{best_risk_recall:.3f}" if best_risk_recall is not None else "—",
    )

    st.markdown("### Signature value layers")
    layer_cols = st.columns(4)
    layer_cards = [
        ("Decision workflow", "The dashboard is structured around how an academic adviser would use evidence: audit → train → explain → review → intervene."),
        ("Interactive front-end", "Navigation, story angle, threshold sliders and intervention controls change the displayed evidence rather than showing static charts."),
        ("Responsible AI", "The project openly separates prediction from decision-making and highlights missed-risk cases, fairness and validity."),
        ("BTEC evidence", "Outputs are exportable and aligned with methodology, analysis, discussion, validity and viva defence."),
    ]
    for col, (title, text) in zip(layer_cols, layer_cards):
        with col:
            st.markdown(f'<div class="executive-card"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    left, right = st.columns([1.18, 1])
    with left:
        st.markdown(
            """
            <div class="panel">
                <h3>Research design</h3>
                <p><b>Problem.</b> A high pass-class F1 can hide poor detection of students who are actually at risk.</p>
                <p><b>Design response.</b> Hyperparameters are selected with <b>Macro-F1</b>, while risk recall, F1-Risk, balanced accuracy, MCC and calibration are reported separately.</p>
                <p><b>Leakage control.</b> Preprocessing is fitted inside every cross-validation fold. G1, G2 and G3 are excluded from model inputs.</p>
                <p><b>Responsible-use boundary.</b> Outputs are support signals, not automatic sanctions or causal conclusions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="panel"><h3>Current experiment</h3>', unsafe_allow_html=True)
        if bundle:
            preview = format_result_table(bundle)[
                ["Rank", "Model", "Macro-F1", "Risk Recall", "AUC-ROC"]
            ].head(5)
            st.dataframe(arrow_safe_df(preview), width="stretch", hide_index=True)
            claim_title, claim_text = performance_statement(bundle)
            st.markdown(
                f'<div class="callout callout-success"><b>{claim_title}</b><br>{claim_text}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "No experiment is stored in this browser session. Open Model Training and run Quick mode first."
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### End-to-end workflow")
    workflow = pd.DataFrame(
        {
            "Stage": [
                "1. Data audit",
                "2. Leakage-safe preprocessing",
                "3. Nested model selection",
                "4. Reliability and error analysis",
                "5. Fairness and explainability",
                "6. Human-reviewed prediction",
            ],
            "Evidence produced": [
                "Missingness, duplicates, class balance and dataset fingerprint",
                "Imputation, scaling and one-hot encoding inside each fold",
                "Baseline comparison, fold metrics and tuned pipelines",
                "Confidence intervals, calibration, confusion matrix and error profiles",
                "Subgroup metrics, disparity warnings and SHAP explanations",
                "Risk probability, threshold control and responsible-use notice",
            ],
        }
    )
    st.dataframe(arrow_safe_df(workflow), width="stretch", hide_index=True)

# 📊 EDA
# ════════════════════════════════════════
elif page == "📊 EDA & Data Audit":
    render_page_header(
        "DATA UNDERSTANDING",
        "Exploratory Data Analysis",
        "Inspect class balance, distributions, correlations and data-quality risks before modelling.",
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Distribution", "🔗 Correlation", "📦 Boxplots", "📋 Dataset", "✅ Data Quality"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            counts = df["target"].value_counts().reindex([0, 1], fill_value=0)
            class_df = pd.DataFrame(
                {"Class": ["Fail", "Pass"], "Count": counts.values}
            )
            fig = px.bar(
                class_df,
                x="Class",
                y="Count",
                color="Class",
                text="Count",
                title="Pass vs Fail Distribution",
                color_discrete_map={"Fail": "#E74C3C", "Pass": "#2ECC71"},
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, width="stretch")

        with col2:
            if "G3" in df.columns:
                fig = px.histogram(
                    df,
                    x="G3",
                    nbins=20,
                    title="Final Grade (G3) Distribution",
                )
                fig.add_vline(
                    x=10,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Pass threshold = 10",
                )
                st.plotly_chart(fig, width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            if "absences" in df.columns:
                fig = px.histogram(
                    df,
                    x="absences",
                    nbins=30,
                    title="Absences Distribution",
                )
                st.plotly_chart(fig, width="stretch")
        with col2:
            pie_df = pd.DataFrame(
                {
                    "Class": df["target"].map({0: "Fail", 1: "Pass"}),
                }
            )
            fig = px.pie(
                pie_df,
                names="Class",
                title="Class Balance",
                color="Class",
                color_discrete_map={"Fail": "#E74C3C", "Pass": "#2ECC71"},
            )
            st.plotly_chart(fig, width="stretch")

    with tab2:
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.4,
            annot_kws={"size": 7},
            ax=ax,
        )
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        available = [
            col
            for col in [
                "studytime",
                "failures",
                "absences",
                "Medu",
                "Fedu",
                "famrel",
                "freetime",
            ]
            if col in df.columns
        ]
        feature = st.selectbox("Select a feature", available)
        y_axis = "G3" if "G3" in df.columns else "target"
        plot_df = df.copy()
        plot_df["Class"] = plot_df["target"].map({0: "Fail", 1: "Pass"})
        fig = px.box(
            plot_df,
            x=feature,
            y=y_axis,
            color="Class",
            title=f"{feature} vs {y_axis}",
            color_discrete_map={"Fail": "#E74C3C", "Pass": "#2ECC71"},
        )
        st.plotly_chart(fig, width="stretch")

    with tab4:
        st.dataframe(arrow_safe_df(df.head(50)), width="stretch", hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing cells", int(df.isna().sum().sum()))
        st.download_button(
            "📥 Dataset CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="student_dataset_with_target.csv",
            mime="text/csv",
            width="stretch",
        )

    with tab5:
        audit = quality_audit(df)
        st.dataframe(arrow_safe_df(audit), width="stretch", hide_index=True)
        class_counts = df["target"].value_counts().reindex([0, 1], fill_value=0)
        minority_share = float(class_counts.min() / class_counts.sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Missing cells", int(df.isna().sum().sum()))
        c2.metric("Duplicate rows", int(df.duplicated().sum()))
        c3.metric("Minority-class share", f"{minority_share:.1%}")
        c4.metric("Dataset fingerprint", dataset_fingerprint(df))

        if minority_share < 0.25:
            st.warning(
                "Class imbalance is material. Accuracy alone is not sufficient; Macro-F1, risk recall, balanced accuracy and MCC are prioritised."
            )
        else:
            st.success("Class balance is acceptable for the current prototype, while balanced metrics are still reported.")

        missing = (
            df.isna().mean().mul(100).sort_values(ascending=False).rename("Missing %").reset_index()
        )
        missing.columns = ["Feature", "Missing %"]
        missing = missing[missing["Missing %"] > 0]
        if not missing.empty:
            fig = px.bar(missing, x="Feature", y="Missing %", title="Missingness by feature")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No missing values were detected in the source file.")

# ════════════════════════════════════════
# ⚡ OPTUNA
# ════════════════════════════════════════
elif page == "⚡ Optuna Optimization":
    st.title("⚡ Optuna Hyperparameter Optimization")
    st.divider()
    st.info(
        "Preprocessing is fitted independently inside each cross-validation fold, "
        "preventing validation data from leaking into training."
    )

    model_choice = st.selectbox(
        "Select a model",
        ["Logistic Regression", "Random Forest", "Gradient Boosting"],
    )
    n_trials = st.slider("Number of trials", 10, 100, 30, 5)

    if st.button("🚀 Run Optuna optimisation", type="primary", width="stretch"):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        progress = st.progress(0)
        status = st.empty()

        def objective(trial):
            if model_choice == "Logistic Regression":
                params = {
                    "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
                    "l1_ratio": trial.suggest_categorical("l1_ratio", [0.0, 1.0]),
                }
            elif model_choice == "Random Forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 80, 350, step=10),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                }
            else:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=10),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.3, log=True
                    ),
                    "max_depth": trial.suggest_int("max_depth", 2, 6),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                }

            estimator = build_optuna_estimator(model_choice, params)
            pipe = Pipeline(
                [
                    ("pre", clone(preprocessor)),
                    ("clf", estimator),
                ]
            )
            scores = cross_val_score(
                pipe,
                X,
                y,
                cv=cv,
                scoring="f1_macro",
                n_jobs=1,
                error_score="raise",
            )
            return float(scores.mean())

        def callback(study, trial):
            done = trial.number + 1
            progress.progress(min(done / n_trials, 1.0))
            status.text(
                f"Trial {done}/{n_trials} — Best Macro-F1: {study.best_value:.4f}"
            )

        sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        try:
            study.optimize(objective, n_trials=n_trials, callbacks=[callback], n_jobs=1)
        except Exception as exc:
            st.error(f"Optuna optimisation failed: {exc}")
        else:
            best_estimator = build_optuna_estimator(model_choice, study.best_params)
            best_pipe = Pipeline(
                [
                    ("pre", clone(preprocessor)),
                    ("clf", best_estimator),
                ]
            )
            best_pipe.fit(X, y)
            st.session_state.setdefault("optuna_models", {})[model_choice] = best_pipe
            st.session_state["last_optuna_study"] = study

            progress.progress(1.0)
            status.empty()
            st.success("Optuna optimisation completed.")
            c1, c2 = st.columns(2)
            c1.metric("🏆 Best CV Macro-F1", f"{study.best_value:.4f}")
            c2.metric("Trials", len(study.trials))
            st.json(study.best_params)

            trials_df = study.trials_dataframe(
                attrs=("number", "value", "params", "state")
            )
            st.dataframe(arrow_safe_df(trials_df), width="stretch", hide_index=True)

            history = pd.DataFrame(
                {
                    "Trial": [t.number + 1 for t in study.trials],
                    "Macro-F1": [t.value for t in study.trials],
                }
            )
            fig = px.line(
                history,
                x="Trial",
                y="Macro-F1",
                markers=True,
                title="Optuna Trial History",
            )
            fig.add_hline(
                y=study.best_value,
                line_dash="dash",
                annotation_text="Best",
            )
            st.plotly_chart(fig, width="stretch")

            try:
                importance = optuna.importance.get_param_importances(study)
                imp_df = pd.DataFrame(
                    {"Parameter": importance.keys(), "Importance": importance.values()}
                ).sort_values("Importance")
                fig = px.bar(
                    imp_df,
                    x="Importance",
                    y="Parameter",
                    orientation="h",
                    title="Hyperparameter Importance",
                )
                st.plotly_chart(fig, width="stretch")
            except Exception:
                st.caption("There are not enough completed trials to estimate parameter importance reliably.")

# ════════════════════════════════════════
# 🤖 MODEL TRAINING
# ════════════════════════════════════════
elif page == "🤖 Model Training":
    render_page_header(
        "MODEL DEVELOPMENT",
        "Nested Cross-Validation Experiment",
        "Tune inside the inner folds, evaluate on untouched outer folds, then fit a final reproducible pipeline.",
    )

    mode = st.radio(
        "Computation mode",
        ["⚡ Quick — 3 outer × 3 inner", "🔬 Full — 5 outer × 5 inner"],
        horizontal=True,
        help="Quick is suitable for live demonstration. Full provides the stronger final experiment but takes longer.",
    )
    if mode.startswith("⚡"):
        outer_splits, inner_splits = 3, 3
        st.info(
            "Quick mode: six classifiers plus a baseline. Use this to verify the complete application before the viva."
        )
    else:
        outer_splits, inner_splits = 5, 5
        st.warning(
            "Full mode performs a substantially larger nested search. Run it once for final evidence and keep the exported results."
        )

    st.markdown(
        """
        <div class="callout">
            <b>Methodological correction:</b> model selection uses Macro-F1 rather than pass-class F1.
            This prevents the majority Pass class from making a weak early-warning model appear strong.
            Risk Recall and F1-Risk are reported as separate decision-relevant metrics.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚀 Train and evaluate all models", type="primary", width="stretch"):
        progress = st.progress(0)
        status = st.empty()

        def update_progress(name, model_idx, models_count, fold_idx, folds_count):
            completed = (model_idx - 1) * folds_count + (fold_idx - 1)
            total = models_count * folds_count
            progress.progress(min(completed / total, 0.99))
            status.text(
                f"{name}: outer fold {fold_idx}/{folds_count} "
                f"({model_idx}/{models_count} models)"
            )

        try:
            trained = train_models(
                X,
                y,
                preprocessor,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                progress_callback=update_progress,
            )
        except Exception as exc:
            st.error(f"Model training failed: {exc}")
        else:
            st.session_state["training_bundle"] = trained
            bundle = trained
            progress.progress(1.0)
            status.empty()
            st.success("Nested cross-validation completed and final pipelines were fitted.")

    bundle = st.session_state.get("training_bundle")
    if bundle:
        best_name = bundle["best_model_name"]
        baseline = "Dummy Baseline"
        macro_uplift = metric_mean(bundle, best_name, "Macro-F1") - metric_mean(
            bundle, baseline, "Macro-F1"
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected model", best_name)
        c2.metric("Macro-F1", f"{metric_mean(bundle, best_name, 'Macro-F1'):.3f}")
        c3.metric("Risk Recall", f"{metric_mean(bundle, best_name, 'Risk Recall'):.3f}")
        c4.metric("Baseline uplift", f"+{macro_uplift:.3f}")

        claim_title, claim_text = performance_statement(bundle)
        st.markdown(
            f'<div class="callout callout-success"><b>{claim_title}</b><br>{claim_text}</div>',
            unsafe_allow_html=True,
        )

        st.subheader("Fold-by-fold Macro-F1")
        fold_df = bundle["fold_metrics"].copy()
        fold_df["Model"] = fold_df["Model"].astype(str)
        fold_df["Fold"] = pd.to_numeric(fold_df["Fold"], errors="coerce").astype("Int64")
        fold_df["Macro-F1"] = pd.to_numeric(
            fold_df["Macro-F1"], errors="coerce"
        ).astype(float)
        fold_df = fold_df.dropna(subset=["Fold", "Macro-F1"])
        fig = px.line(
            fold_df,
            x="Fold",
            y="Macro-F1",
            color="Model",
            markers=True,
            title="Outer-fold Macro-F1 stability",
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, width="stretch")

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "📦 Download selected model",
                data=model_to_bytes(bundle["best_pipes"][best_name]),
                file_name="best_student_risk_model.joblib",
                mime="application/octet-stream",
                width="stretch",
            )
        with d2:
            st.download_button(
                "🗂️ Download complete evidence pack",
                data=build_evidence_pack(bundle, df, X),
                file_name="btec_ml_evidence_pack.zip",
                mime="application/zip",
                width="stretch",
            )

# 📈 RESULTS
# ════════════════════════════════════════
elif page == "📈 Results":
    render_page_header(
        "EVIDENCE AND FINDINGS",
        "Model Evaluation Dashboard",
        "Compare models against a baseline, inspect risk detection, test stability and export auditable evidence.",
    )

    if not bundle:
        st.warning("Run the nested cross-validation experiment in Model Training first.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "🏆 Leaderboard",
            "📊 Metric comparison",
            "📉 ROC & risk PR",
            "🧩 Threshold & confusion",
            "🎯 Calibration",
            "🧪 Confidence & statistics",
        ]
    )

    model_names = list(bundle["results"].keys())
    best_name = bundle["best_model_name"]
    primary_metric = bundle.get("primary_metric", PRIMARY_METRIC)

    with tab1:
        leaderboard = format_result_table(bundle)
        st.dataframe(arrow_safe_df(leaderboard), width="stretch", hide_index=True)

        baseline_macro = metric_mean(bundle, "Dummy Baseline", "Macro-F1")
        best_macro = metric_mean(bundle, best_name, "Macro-F1")
        uplift = best_macro - baseline_macro
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected model", best_name)
        c2.metric("Macro-F1", f"{best_macro:.3f} ± {metric_std(bundle, best_name, 'Macro-F1'):.3f}")
        c3.metric("Risk Recall", f"{metric_mean(bundle, best_name, 'Risk Recall'):.3f}")
        c4.metric("Uplift vs baseline", f"+{uplift:.3f}")

        claim_title, claim_text = performance_statement(bundle)
        st.markdown(
            f'<div class="callout callout-success"><b>{claim_title}</b><br>{claim_text}</div>',
            unsafe_allow_html=True,
        )

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "📥 Leaderboard CSV",
                data=leaderboard.to_csv(index=False).encode("utf-8"),
                file_name="model_leaderboard.csv",
                mime="text/csv",
                width="stretch",
            )
        with d2:
            st.download_button(
                "📄 Fold metrics CSV",
                data=bundle["fold_metrics"].to_csv(index=False).encode("utf-8"),
                file_name="fold_metrics.csv",
                mime="text/csv",
                width="stretch",
            )
        with d3:
            st.download_button(
                "🗂️ Evidence pack ZIP",
                data=build_evidence_pack(bundle, df, X),
                file_name="btec_ml_evidence_pack.zip",
                mime="application/zip",
                width="stretch",
            )

    with tab2:
        available_metrics = [
            "Macro-F1",
            "Risk Recall",
            "F1-Risk",
            "Balanced Accuracy",
            "MCC",
            "AUC-ROC",
            "PR-AUC-Risk",
            "Missed Risk Rate",
            "False Alarm Rate",
            "F1-Pass",
        ]
        metric = st.selectbox("Metric", available_metrics)
        metric_df = pd.DataFrame(
            {
                "Model": model_names,
                "Mean": [metric_mean(bundle, name, metric) for name in model_names],
                "Std": [metric_std(bundle, name, metric) for name in model_names],
            }
        )
        lower_is_better = metric in {"Missed Risk Rate", "False Alarm Rate"}
        metric_df = metric_df.sort_values("Mean", ascending=lower_is_better)

        fig = go.Figure(
            go.Bar(
                x=metric_df["Model"],
                y=metric_df["Mean"],
                error_y={"type": "data", "array": metric_df["Std"]},
                text=[f"{value:.3f}" for value in metric_df["Mean"]],
                textposition="outside",
            )
        )
        fig.update_layout(title=f"{metric} across candidate models")
        if metric != "MCC":
            fig.update_yaxes(range=[0, 1.08])
        else:
            fig.update_yaxes(range=[-1, 1])
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### Multi-criteria view")
        radar_metrics = ["Macro-F1", "Risk Recall", "Balanced Accuracy", "AUC-ROC", "PR-AUC-Risk"]
        top3 = format_result_table(bundle)["Model"].head(3).tolist()
        radar = go.Figure()
        for name in top3:
            values = [metric_mean(bundle, name, item) for item in radar_metrics]
            radar.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=radar_metrics + [radar_metrics[0]],
                    fill="toself",
                    name=name,
                )
            )
        radar.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
            title="Top three models across balanced and risk-focused criteria",
        )
        st.plotly_chart(radar, width="stretch")

    with tab3:
        selected_models = st.multiselect(
            "Models",
            model_names,
            default=[name for name in model_names if name != "Dummy Baseline"],
        )
        roc_fig = go.Figure()
        pr_fig = go.Figure()
        risk_prevalence = float((1 - y).mean())

        for name in selected_models:
            pred_data = bundle["oof"][name]
            y_true = np.asarray(pred_data["y_true"])
            y_prob_pass = np.asarray(pred_data["y_prob"])

            fpr, tpr, _ = roc_curve(y_true, y_prob_pass)
            roc_value = auc(fpr, tpr)
            roc_fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{name} (AUC={roc_value:.3f})",
                )
            )

            risk_true = 1 - y_true
            risk_prob = 1 - y_prob_pass
            precision, recall, _ = precision_recall_curve(risk_true, risk_prob)
            pr_value = average_precision_score(risk_true, risk_prob)
            pr_fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=f"{name} (AP={pr_value:.3f})",
                )
            )

        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random", line={"dash": "dash"}
            )
        )
        pr_fig.add_hline(
            y=risk_prevalence,
            line_dash="dash",
            annotation_text=f"Risk prevalence = {risk_prevalence:.2f}",
        )
        roc_fig.update_layout(
            title="Out-of-fold ROC curves (Pass probability)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        pr_fig.update_layout(
            title="Out-of-fold Precision–Recall curves for the at-risk class",
            xaxis_title="Risk Recall",
            yaxis_title="Risk Precision",
        )
        st.plotly_chart(roc_fig, width="stretch")
        st.plotly_chart(pr_fig, width="stretch")
        st.caption(
            "Risk PR is central because the academic use-case is finding students with G3 < 10, not merely predicting the majority Pass class."
        )

    with tab4:
        selected = st.selectbox("Model", model_names, index=model_names.index(best_name))
        threshold = st.slider(
            "Pass-probability threshold",
            0.10,
            0.90,
            0.50,
            0.05,
            help="A higher threshold flags more students as at risk; this can improve risk recall while increasing false alarms.",
        )
        pred_data = bundle["oof"][selected]
        metrics = oof_metrics(pred_data["y_true"], pred_data["y_prob"], threshold)
        y_pred_threshold = (np.asarray(pred_data["y_prob"]) >= threshold).astype(int)
        cm = confusion_matrix(pred_data["y_true"], y_pred_threshold, labels=[0, 1])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Recall", f"{metrics['Risk Recall']:.3f}")
        c2.metric("Missed Risk Rate", f"{metrics['Missed Risk Rate']:.3f}")
        c3.metric("False Alarm Rate", f"{metrics['False Alarm Rate']:.3f}")
        c4.metric("Macro-F1", f"{metrics['Macro-F1']:.3f}")

        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted at risk", "Predicted pass"],
            y=["Actual at risk", "Actual pass"],
            title=f"{selected} — out-of-fold confusion matrix",
            labels={"x": "Prediction", "y": "Actual", "color": "Students"},
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            """
            <div class="callout callout-risk">
                <b>Critical error:</b> the top-right cell contains actual at-risk students predicted as Pass.
                In this target coding, that is a conventional false positive but a practical <b>missed-risk case</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Cost-sensitive threshold analysis")
        st.caption(
            "This analysis makes the operational trade-off explicit. Assigning a larger cost to missed-risk cases recommends a threshold that prioritises early intervention."
        )
        cost_left, cost_right = st.columns(2)
        missed_cost = cost_left.slider(
            "Relative cost of one missed-risk case",
            1.0,
            20.0,
            5.0,
            1.0,
            key="missed_risk_cost",
        )
        false_alarm_cost = cost_right.slider(
            "Relative cost of one false alarm",
            1.0,
            20.0,
            1.0,
            1.0,
            key="false_alarm_cost",
        )
        cost_df = threshold_cost_analysis(
            pred_data["y_true"],
            pred_data["y_prob"],
            missed_cost,
            false_alarm_cost,
        )
        recommended = cost_df.loc[cost_df["Weighted cost"].idxmin()]
        cost_fig = px.line(
            cost_df,
            x="Pass threshold",
            y="Cost per student",
            markers=True,
            title="Weighted operational cost across thresholds",
        )
        cost_fig.add_vline(
            x=float(recommended["Pass threshold"]),
            line_dash="dash",
            annotation_text=f"Recommended = {recommended['Pass threshold']:.2f}",
        )
        st.plotly_chart(cost_fig, width="stretch")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Recommended threshold", f"{recommended['Pass threshold']:.2f}")
        rc2.metric("Risk recall", f"{recommended['Risk Recall']:.3f}")
        rc3.metric("Missed-risk cases", int(recommended["Missed-risk cases"]))
        rc4.metric("False alarms", int(recommended["False alarms"]))
        st.download_button(
            "📥 Download threshold-cost analysis",
            data=cost_df.to_csv(index=False).encode("utf-8"),
            file_name=f"threshold_cost_{selected.replace(' ', '_')}.csv",
            mime="text/csv",
            width="stretch",
        )

    with tab5:
        selected = st.selectbox(
            "Calibration model",
            model_names,
            index=model_names.index(best_name),
            key="calibration_model",
        )
        pred_data = bundle["oof"][selected]
        y_true = np.asarray(pred_data["y_true"])
        y_prob = np.asarray(pred_data["y_prob"])
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=8, strategy="quantile"
        )
        brier = brier_score_loss(y_true, y_prob)

        calibration_fig = go.Figure()
        calibration_fig.add_trace(
            go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name=selected)
        )
        calibration_fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line={"dash": "dash"}
            )
        )
        calibration_fig.update_layout(
            title=f"Calibration curve — {selected}",
            xaxis_title="Predicted Pass probability",
            yaxis_title="Observed Pass rate",
            xaxis={"range": [0, 1]},
            yaxis={"range": [0, 1]},
        )
        st.plotly_chart(calibration_fig, width="stretch")
        c1, c2 = st.columns(2)
        c1.metric("Brier score", f"{brier:.4f}")
        c2.metric("Mean predicted Pass probability", f"{y_prob.mean():.3f}")
        st.caption(
            "Calibration assesses whether an 80% prediction behaves like an 80% event rate. It does not prove generalisation to a new university."
        )

    with tab6:
        selected = st.selectbox(
            "Model for bootstrap confidence intervals",
            model_names,
            index=model_names.index(best_name),
            key="ci_model",
        )
        threshold = st.slider(
            "CI threshold",
            0.10,
            0.90,
            0.50,
            0.05,
            key="ci_threshold",
        )
        pred_data = bundle["oof"][selected]
        ci_df = bootstrap_confidence_intervals(
            pred_data["y_true"], pred_data["y_prob"], threshold, iterations=500
        )
        display_ci = ci_df.copy()
        for column in ["Estimate", "95% CI lower", "95% CI upper"]:
            display_ci[column] = display_ci[column].round(3)
        st.dataframe(arrow_safe_df(display_ci), width="stretch", hide_index=True)

        st.markdown("#### Paired fold comparison")
        left, right = st.columns(2)
        model_a = left.selectbox("Model A", model_names, index=model_names.index(best_name))
        model_b_options = [name for name in model_names if name != model_a]
        default_b = model_b_options.index("Dummy Baseline") if "Dummy Baseline" in model_b_options else 0
        model_b = right.selectbox("Model B", model_b_options, index=default_b)
        a_scores = bundle["score_arrays"][model_a]
        b_scores = bundle["score_arrays"][model_b]
        comparison_df = pd.DataFrame(
            {
                "Outer fold": np.arange(1, len(a_scores) + 1),
                model_a: a_scores,
                model_b: b_scores,
                "Difference": a_scores - b_scores,
            }
        )
        st.dataframe(arrow_safe_df(comparison_df.round(4)), width="stretch", hide_index=True)
        try:
            statistic, p_value = wilcoxon(a_scores, b_scores)
            c1, c2, c3 = st.columns(3)
            c1.metric("Wilcoxon statistic", f"{statistic:.4f}")
            c2.metric("p-value", f"{p_value:.4f}")
            c3.metric("Mean difference", f"{(a_scores - b_scores).mean():.4f}")
            if p_value < 0.05:
                st.success("The paired outer-fold difference is statistically significant at α=0.05.")
            else:
                st.info(
                    "No statistically significant difference was detected. With only 3 or 5 outer folds, statistical power is limited; this is not proof of equivalence."
                )
        except ValueError as exc:
            st.warning(f"Wilcoxon test could not be calculated: {exc}")

        st.markdown("#### Omnibus and multiple-comparison analysis")
        st.caption(
            "The Friedman test assesses whether matched outer-fold scores differ across all models. Pairwise Wilcoxon tests are then adjusted with the Holm procedure to reduce false discoveries."
        )
        comparable_names = list(bundle["score_arrays"].keys())
        comparable_arrays = [
            np.asarray(bundle["score_arrays"][name], dtype=float)
            for name in comparable_names
        ]
        try:
            friedman_stat, friedman_p = friedmanchisquare(*comparable_arrays)
            fc1, fc2 = st.columns(2)
            fc1.metric("Friedman statistic", f"{friedman_stat:.4f}")
            fc2.metric("Friedman p-value", f"{friedman_p:.4f}")
            if friedman_p < 0.05:
                st.success(
                    "The omnibus test indicates that at least one model differs across the matched outer folds."
                )
            else:
                st.info(
                    "The omnibus test did not detect a difference. With only 3 or 5 outer folds, this result has low statistical power and must not be interpreted as proof that all models are equivalent."
                )
        except ValueError as exc:
            st.warning(f"Friedman test could not be calculated: {exc}")

        pairwise_df = pairwise_wilcoxon_results(bundle)
        if not pairwise_df.empty:
            display_pairwise = pairwise_df.copy()
            for column in [
                "Mean difference",
                "Wilcoxon statistic",
                "Raw p-value",
                "Holm-adjusted p-value",
            ]:
                display_pairwise[column] = pd.to_numeric(
                    display_pairwise[column], errors="coerce"
                ).round(4)
            st.dataframe(
                arrow_safe_df(display_pairwise),
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "📥 Download corrected pairwise tests",
                data=pairwise_df.to_csv(index=False).encode("utf-8"),
                file_name="pairwise_wilcoxon_holm.csv",
                mime="text/csv",
                width="stretch",
            )

# ⚖️ FAIRNESS ANALYSIS
# ════════════════════════════════════════
elif page == "⚖️ Fairness Analysis":
    render_page_header(
        "RESPONSIBLE AI",
        "Fairness and Subgroup Performance",
        "Examine whether risk detection and error rates vary across demographic or contextual groups.",
    )
    st.markdown(
        """
        <div class="callout callout-risk">
            Subgroup differences are diagnostic signals, not automatic proof of discrimination.
            Small samples, historical bias and unobserved context must be considered before drawing conclusions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not bundle:
        st.info("Run Quick or Full model training before opening subgroup analysis.")
        st.stop()

    controls = st.columns(4)
    with controls[0]:
        fairness_model = st.selectbox(
            "Model",
            list(bundle["results"].keys()),
            index=list(bundle["results"].keys()).index(bundle["best_model_name"]),
        )
    available_groups = [
        column
        for column in ["sex", "school", "address", "Medu", "Fedu", "age"]
        if column in X.columns
    ]
    with controls[1]:
        group_feature = st.selectbox("Subgroup feature", available_groups)
    with controls[2]:
        threshold = st.slider("Pass threshold", 0.10, 0.90, 0.50, 0.05)
    with controls[3]:
        min_group_size = st.slider("Minimum group size", 5, 30, 10)

    fairness_df = X[[group_feature]].copy()
    if group_feature == "age":
        fairness_df["Group"] = pd.cut(
            pd.to_numeric(fairness_df[group_feature], errors="coerce"),
            bins=[14, 16, 18, 20, 30],
            labels=["15–16", "17–18", "19–20", "21+"],
            include_lowest=True,
        ).astype(str)
    else:
        fairness_df["Group"] = fairness_df[group_feature].astype(str)

    pred_data = bundle["oof"][fairness_model]
    fairness_df["y_true"] = np.asarray(pred_data["y_true"], dtype=int)
    fairness_df["y_prob"] = np.asarray(pred_data["y_prob"], dtype=float)
    fairness_df["y_pred"] = (fairness_df["y_prob"] >= threshold).astype(int)

    rows = []
    for group, part in fairness_df.groupby("Group", dropna=False):
        if len(part) < min_group_size:
            continue
        y_true_group = part["y_true"].to_numpy()
        y_pred_group = part["y_pred"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(
            y_true_group, y_pred_group, labels=[0, 1]
        ).ravel()
        risk_recall = tn / (tn + fp) if (tn + fp) else np.nan
        risk_precision = tn / (tn + fn) if (tn + fn) else np.nan
        missed_risk = fp / (tn + fp) if (tn + fp) else np.nan
        false_alarm = fn / (fn + tp) if (fn + tp) else np.nan
        rows.append(
            {
                "Group": str(group),
                "N": int(len(part)),
                "Actual risk rate": float((y_true_group == 0).mean()),
                "Predicted risk rate": float((y_pred_group == 0).mean()),
                "Risk Recall": float(risk_recall),
                "Risk Precision": float(risk_precision),
                "F1-Risk": float(
                    f1_score(y_true_group, y_pred_group, pos_label=0, zero_division=0)
                ),
                "Macro-F1": float(
                    f1_score(y_true_group, y_pred_group, average="macro", zero_division=0)
                ),
                "Missed Risk Rate": float(missed_risk),
                "False Alarm Rate": float(false_alarm),
            }
        )

    subgroup_results = pd.DataFrame(rows)
    if subgroup_results.empty:
        st.error("No subgroup meets the selected minimum size.")
        st.stop()

    metric_columns = [
        "Actual risk rate",
        "Predicted risk rate",
        "Risk Recall",
        "Risk Precision",
        "F1-Risk",
        "Macro-F1",
        "Missed Risk Rate",
        "False Alarm Rate",
    ]
    fairness_display = subgroup_results.copy()
    fairness_display[metric_columns] = fairness_display[metric_columns].round(3)
    st.dataframe(arrow_safe_df(fairness_display), width="stretch", hide_index=True)

    metric_choice = st.selectbox(
        "Metric to compare",
        ["Risk Recall", "F1-Risk", "Missed Risk Rate", "False Alarm Rate", "Predicted risk rate", "Macro-F1"],
    )
    fairness_fig = px.bar(
        subgroup_results,
        x="Group",
        y=metric_choice,
        text=subgroup_results[metric_choice].round(3),
        title=f"{fairness_model}: {metric_choice} by {feature_label(group_feature)}",
    )
    fairness_fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fairness_fig, width="stretch")

    spread = float(
        subgroup_results[metric_choice].max() - subgroup_results[metric_choice].min()
    )
    largest_group = subgroup_results.sort_values("N", ascending=False).iloc[0]["Group"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Observed spread", f"{spread:.3f}")
    c2.metric("Groups analysed", len(subgroup_results))
    c3.metric("Largest reference group", str(largest_group))

    if spread >= 0.15:
        st.error(
            f"Material subgroup disparity signal: {metric_choice} differs by {spread:.3f}. Investigate sample size, data quality and institutional context before deployment."
        )
    elif spread >= 0.08:
        st.warning(
            f"Moderate subgroup difference detected: {spread:.3f}. Report it as a limitation and inspect confidence intervals before making a fairness claim."
        )
    else:
        st.success(
            f"No large subgroup spread was observed for this metric at the selected threshold ({spread:.3f}). This does not prove fairness."
        )

    st.download_button(
        "📥 Download subgroup evidence",
        data=subgroup_results.to_csv(index=False).encode("utf-8"),
        file_name=f"fairness_{group_feature}_{fairness_model.replace(' ', '_')}.csv",
        mime="text/csv",
        width="stretch",
    )


# ════════════════════════════════════════
# 🧪 VALIDITY & ERROR ANALYSIS
# ════════════════════════════════════════
elif page == "🧪 Validity & Error Analysis":
    render_page_header(
        "DISTINCTION EVIDENCE",
        "Validity, Reliability and Error Analysis",
        "Test the strength of the evidence, inspect critical errors and state what the model cannot establish.",
    )

    if not bundle:
        st.info("Run model training before evaluating validity and error patterns.")
        st.stop()

    control1, control2 = st.columns([1, 1])
    with control1:
        selected_model = st.selectbox(
            "Model",
            list(bundle["results"].keys()),
            index=list(bundle["results"].keys()).index(bundle["best_model_name"]),
            key="validity_model",
        )
    with control2:
        threshold = st.slider(
            "Pass-probability threshold",
            0.10,
            0.90,
            0.50,
            0.05,
            key="validity_threshold",
        )

    pred_data = bundle["oof"][selected_model]
    y_true = np.asarray(pred_data["y_true"], dtype=int)
    y_prob = np.asarray(pred_data["y_prob"], dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = oof_metrics(y_true, y_prob, threshold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Macro-F1", f"{metrics['Macro-F1']:.3f}")
    c2.metric("Risk Recall", f"{metrics['Risk Recall']:.3f}")
    c3.metric("Missed Risk Rate", f"{metrics['Missed Risk Rate']:.3f}")
    c4.metric("MCC", f"{metrics['MCC']:.3f}")

    error_df = X.copy().reset_index(drop=True)
    error_df["Actual"] = np.where(y_true == 0, "At risk", "Pass")
    error_df["Prediction"] = np.where(y_pred == 0, "At risk", "Pass")
    error_df["Pass probability"] = y_prob
    error_df["Risk probability"] = 1 - y_prob
    error_df["Error type"] = np.select(
        [
            (y_true == 0) & (y_pred == 1),
            (y_true == 1) & (y_pred == 0),
            (y_true == y_pred),
        ],
        ["Missed risk", "False alarm", "Correct"],
        default="Unknown",
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🚨 Critical errors",
            "📊 Error profiles",
            "📏 Confidence intervals",
            "✅ Validity assessment",
            "📚 Learning curve",
        ]
    )

    with tab1:
        counts = (
            error_df["Error type"].value_counts().reindex(["Correct", "Missed risk", "False alarm"], fill_value=0).reset_index()
        )
        counts.columns = ["Outcome", "Students"]
        fig = px.bar(
            counts,
            x="Outcome",
            y="Students",
            color="Outcome",
            text="Students",
            title="Out-of-fold prediction outcomes",
            color_discrete_map={
                "Correct": "#0f766e",
                "Missed risk": "#b91c1c",
                "False alarm": "#b45309",
            },
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown(
            """
            <div class="callout callout-risk">
                <b>Missed risk</b> means an actual Fail/at-risk student was predicted as Pass.
                This is the most important operational error for an early-warning system.
                <b>False alarm</b> means a student who actually passed was flagged as at risk.
            </div>
            """,
            unsafe_allow_html=True,
        )

        missed = error_df[error_df["Error type"] == "Missed risk"].sort_values(
            "Pass probability", ascending=False
        )
        st.markdown(f"#### Missed-risk cases ({len(missed)})")
        key_columns = [
            column
            for column in ["school", "sex", "age", "studytime", "failures", "absences", "health"]
            if column in missed.columns
        ]
        display_columns = key_columns + ["Pass probability", "Risk probability"]
        if missed.empty:
            st.success("No missed-risk cases occurred at this threshold.")
        else:
            st.dataframe(
                arrow_safe_df(missed[display_columns].head(30).round(3)),
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "📥 Download missed-risk cases",
                data=missed.to_csv(index=False).encode("utf-8"),
                file_name="missed_risk_cases.csv",
                mime="text/csv",
                width="stretch",
            )

    with tab2:
        numeric_features = [column for column in num_cols if column in error_df.columns]
        selected_feature = st.selectbox(
            "Numeric feature",
            numeric_features,
            format_func=feature_label,
            key="error_feature",
        )
        profile = (
            error_df.groupby("Error type")[selected_feature]
            .agg(["count", "mean", "median", "std"])
            .reset_index()
        )
        st.dataframe(arrow_safe_df(profile.round(3)), width="stretch", hide_index=True)
        fig = px.box(
            error_df,
            x="Error type",
            y=selected_feature,
            color="Error type",
            title=f"{feature_label(selected_feature)} by prediction outcome",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "This is descriptive error analysis. Differences do not establish causation and may reflect class imbalance or correlated variables."
        )

    with tab3:
        ci_df = bootstrap_confidence_intervals(y_true, y_prob, threshold, iterations=700)
        display_ci = ci_df.copy()
        for column in ["Estimate", "95% CI lower", "95% CI upper"]:
            display_ci[column] = display_ci[column].round(3)
        st.dataframe(arrow_safe_df(display_ci), width="stretch", hide_index=True)
        ci_long = ci_df.melt(
            id_vars="Metric",
            value_vars=["95% CI lower", "Estimate", "95% CI upper"],
            var_name="Statistic",
            value_name="Value",
        )
        fig = px.scatter(
            ci_long,
            x="Value",
            y="Metric",
            color="Statistic",
            title="Bootstrap 95% confidence intervals",
        )
        fig.update_xaxes(range=[-0.05, 1.05])
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Confidence intervals quantify sampling uncertainty within this dataset. They do not replace external validation on a different institution or time period."
        )

    with tab4:
        validation_rows = [
            {
                "Validity dimension": "Internal validity",
                "Evidence": "Preprocessing is fitted inside CV folds; outer folds remain untouched during tuning.",
                "Residual limitation": "Observational data and unmeasured confounding prevent causal claims.",
                "RAG": "Green/Amber",
            },
            {
                "Validity dimension": "Construct validity",
                "Evidence": "At-risk status is operationalised as G3 < 10, with prior grades excluded.",
                "Residual limitation": "A single final-grade threshold cannot represent every form of academic risk.",
                "RAG": "Amber",
            },
            {
                "Validity dimension": "Reliability",
                "Evidence": "Fixed random seed, nested CV, fold-level results, confidence intervals and dataset fingerprint.",
                "Residual limitation": "Only 395 records; fold estimates can vary.",
                "RAG": "Amber",
            },
            {
                "Validity dimension": "External validity",
                "Evidence": "None beyond the source dataset.",
                "Residual limitation": "No PDP University or cross-institution validation; direct deployment is not justified.",
                "RAG": "Red",
            },
            {
                "Validity dimension": "Ethical validity",
                "Evidence": "Fairness analysis, human-review requirement and explicit intended-use limits.",
                "Residual limitation": "Subgroup samples are small and sensitive attributes remain present.",
                "RAG": "Amber",
            },
        ]
        st.dataframe(
            arrow_safe_df(pd.DataFrame(validation_rows)),
            width="stretch",
            hide_index=True,
        )
        st.markdown(
            """
            <div class="callout">
                <b>Defensible conclusion:</b> the application demonstrates a technically valid academic prototype with transparent limitations.
                It does not demonstrate production readiness, causality or universal generalisability.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab5:
        st.markdown("#### Learning-curve diagnosis")
        st.caption(
            "A learning curve helps distinguish underfitting from data scarcity. This is an on-demand diagnostic and may take up to a minute on Streamlit Cloud."
        )
        if st.button(
            "Calculate learning curve",
            type="primary",
            key="learning_curve_button",
            width="stretch",
        ):
            with st.spinner("Estimating training and validation performance..."):
                estimator = clone(bundle["best_pipes"][selected_model])
                cv = StratifiedKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=RANDOM_STATE,
                )
                sizes, train_scores, validation_scores = learning_curve(
                    estimator,
                    X,
                    y,
                    cv=cv,
                    scoring="f1_macro",
                    train_sizes=np.linspace(0.2, 1.0, 5),
                    n_jobs=1,
                    shuffle=True,
                    random_state=RANDOM_STATE,
                    error_score="raise",
                )
            learning_df = pd.DataFrame(
                {
                    "Training examples": sizes,
                    "Training Macro-F1": train_scores.mean(axis=1),
                    "Validation Macro-F1": validation_scores.mean(axis=1),
                    "Validation standard deviation": validation_scores.std(axis=1, ddof=1),
                }
            )
            long_df = learning_df.melt(
                id_vars="Training examples",
                value_vars=["Training Macro-F1", "Validation Macro-F1"],
                var_name="Series",
                value_name="Macro-F1",
            )
            curve = px.line(
                long_df,
                x="Training examples",
                y="Macro-F1",
                color="Series",
                markers=True,
                title=f"Learning curve — {selected_model}",
            )
            curve.update_yaxes(range=[0, 1.05])
            st.plotly_chart(curve, width="stretch")
            st.dataframe(
                arrow_safe_df(learning_df.round(4)),
                width="stretch",
                hide_index=True,
            )
            final_gap = float(
                learning_df.iloc[-1]["Training Macro-F1"]
                - learning_df.iloc[-1]["Validation Macro-F1"]
            )
            if final_gap >= 0.15:
                st.warning(
                    f"The final train-validation gap is {final_gap:.3f}, which suggests overfitting or high variance. More data, stronger regularisation or a simpler model should be considered."
                )
            elif learning_df.iloc[-1]["Validation Macro-F1"] < 0.60:
                st.info(
                    "Training and validation performance are both limited, which suggests underfitting, weak predictors or an outcome that is difficult to predict without prior grades."
                )
            else:
                st.success(
                    "The final learning-curve gap is relatively controlled. External validation is still required before deployment."
                )
            st.download_button(
                "📥 Download learning-curve data",
                data=learning_df.to_csv(index=False).encode("utf-8"),
                file_name=f"learning_curve_{selected_model.replace(' ', '_')}.csv",
                mime="text/csv",
                width="stretch",
            )


# 🔬 SHAP
# ════════════════════════════════════════
elif page == "🔬 SHAP Values":
    render_page_header(
        "EXPLAINABLE AI",
        "Global and Individual SHAP Explanations",
        "Show which transformed features influence the model overall and why a selected student receives a particular score.",
    )
    st.info(
        "SHAP explanations are generated with Gradient Boosting. They describe model behaviour, not causal effects on academic performance."
    )

    sample_size = st.slider(
        "SHAP sample size", 30, min(200, len(X)), min(100, len(X)), 10
    )

    if st.button("🔬 Calculate SHAP explanations", type="primary", width="stretch"):
        import shap

        with st.spinner("Calculating SHAP values..."):
            if bundle and "Gradient Boosting" in bundle["best_pipes"]:
                shap_pipe = bundle["best_pipes"]["Gradient Boosting"]
                shap_source = "Nested-CV tuned Gradient Boosting pipeline"
            else:
                shap_pipe = Pipeline(
                    [
                        ("pre", clone(preprocessor)),
                        (
                            "clf",
                            GradientBoostingClassifier(
                                n_estimators=120,
                                learning_rate=0.05,
                                max_depth=2,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                )
                shap_pipe.fit(X, y)
                shap_source = "Fallback Gradient Boosting pipeline"

            X_sample = X.sample(
                n=min(sample_size, len(X)), random_state=RANDOM_STATE
            )
            pre = shap_pipe.named_steps["pre"]
            clf = shap_pipe.named_steps["clf"]
            X_processed = np.asarray(pre.transform(X_sample))
            feature_names = pre.get_feature_names_out()

            background = X_processed[: min(100, len(X_processed))]
            explainer = shap.Explainer(
                clf,
                background,
                feature_names=feature_names,
            )
            explanation = normalize_shap_explanation(
                explainer(X_processed), feature_names
            )

            st.session_state["shap_explanation"] = explanation
            st.session_state["shap_rows"] = X_sample.reset_index(drop=True)
            st.session_state["shap_source"] = shap_source

        st.success("SHAP explanations were calculated.")

    if "shap_explanation" in st.session_state:
        import shap

        explanation = st.session_state["shap_explanation"]
        shap_rows = st.session_state["shap_rows"]
        st.caption(st.session_state.get("shap_source", "Gradient Boosting pipeline"))

        mean_abs = np.abs(np.asarray(explanation.values)).mean(axis=0)
        importance_df = pd.DataFrame(
            {
                "Transformed feature": explanation.feature_names,
                "Mean absolute SHAP": mean_abs,
            }
        ).sort_values("Mean absolute SHAP", ascending=False)

        tab1, tab2, tab3 = st.tabs(
            ["📊 Global importance", "🐝 Direction and spread", "👤 Individual explanation"]
        )
        with tab1:
            plt.figure(figsize=(10, 7))
            shap.plots.bar(explanation, max_display=15, show=False)
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(
                arrow_safe_df(importance_df.head(20).round(5)),
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "📥 Download SHAP importance",
                data=importance_df.to_csv(index=False).encode("utf-8"),
                file_name="shap_global_importance.csv",
                mime="text/csv",
                width="stretch",
            )

        with tab2:
            plt.figure(figsize=(10, 7))
            shap.plots.beeswarm(explanation, max_display=15, show=False)
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "Positive SHAP values push the model towards Pass; negative values push it towards the at-risk class. Correlation is not causation."
            )

        with tab3:
            row_number = st.slider("Sample row", 1, len(shap_rows), 1)
            st.dataframe(
                arrow_safe_df(shap_rows.iloc[[row_number - 1]]),
                width="stretch",
                hide_index=True,
            )
            row_explanation = explanation[row_number - 1]
            local_df = pd.DataFrame(
                {
                    "Transformed feature": row_explanation.feature_names,
                    "SHAP contribution": np.asarray(row_explanation.values),
                    "Feature value": np.asarray(row_explanation.data),
                }
            )
            local_df["Absolute contribution"] = local_df["SHAP contribution"].abs()
            local_df = local_df.sort_values("Absolute contribution", ascending=False)

            plt.figure(figsize=(10, 7))
            shap.plots.waterfall(row_explanation, max_display=15, show=False)
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)

            positive = local_df[local_df["SHAP contribution"] > 0].head(5)
            negative = local_df[local_df["SHAP contribution"] < 0].head(5)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Strongest factors towards Pass")
                st.dataframe(
                    arrow_safe_df(positive[["Transformed feature", "SHAP contribution"]].round(4)),
                    width="stretch",
                    hide_index=True,
                )
            with c2:
                st.markdown("#### Strongest factors towards risk")
                st.dataframe(
                    arrow_safe_df(negative[["Transformed feature", "SHAP contribution"]].round(4)),
                    width="stretch",
                    hide_index=True,
                )


# 🔍 PREDICTION
# ════════════════════════════════════════
elif page == "🔍 Prediction":
    render_page_header(
        "DECISION SUPPORT",
        "Individual Student Risk Estimate",
        "Generate a transparent support signal from the trained pipeline. This is not an automated academic decision.",
    )
    st.markdown(
        """
        <div class="callout callout-risk">
            <b>Responsible-use rule:</b> use the result to trigger human review and support, never as the sole basis for punishment, exclusion or automatic rejection.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if bundle:
        prediction_model_name = bundle["best_model_name"]
        prediction_model = bundle["best_pipes"][prediction_model_name]
        model_source = "Nested-CV tuned final pipeline"
    elif st.session_state.get("optuna_models"):
        prediction_model_name = next(iter(st.session_state["optuna_models"]))
        prediction_model = st.session_state["optuna_models"][prediction_model_name]
        model_source = "Optuna-trained pipeline"
    else:
        prediction_model_name = "Logistic Regression (default)"
        prediction_model = get_cached_default_model(X, y)
        model_source = "Fallback pipeline; comparative evaluation has not been run"

    info1, info2 = st.columns([2, 1])
    info1.info(f"Model: **{prediction_model_name}** · {model_source}")
    pass_threshold = info2.slider(
        "Pass threshold",
        0.10,
        0.90,
        0.50,
        0.05,
        help="A higher threshold flags more students as at risk.",
    )

    inputs = {}
    with st.form("prediction_form"):
        st.markdown("#### Student profile")
        columns = st.columns(3)
        for idx, feature in enumerate(X.columns):
            container = columns[idx % 3]
            series = X[feature].dropna()
            label = feature_label(feature)

            with container:
                if feature in cat_cols:
                    options = series.unique().tolist()
                    mode = series.mode().iloc[0] if not series.mode().empty else options[0]
                    default_index = options.index(mode) if mode in options else 0
                    inputs[feature] = st.selectbox(
                        label,
                        options,
                        index=default_index,
                        key=f"pred_{feature}",
                        help=f"Dataset field: {feature}",
                    )
                else:
                    numeric = pd.to_numeric(series, errors="coerce").dropna()
                    minimum = float(numeric.min())
                    maximum = float(numeric.max())
                    median = float(numeric.median())
                    is_integer = pd.api.types.is_integer_dtype(X[feature].dtype)
                    unique_values = sorted(numeric.unique().tolist())

                    if is_integer and len(unique_values) <= 20:
                        values = [int(value) for value in unique_values]
                        default = int(round(median))
                        default_index = (
                            values.index(default)
                            if default in values
                            else min(range(len(values)), key=lambda i: abs(values[i] - default))
                        )
                        inputs[feature] = st.selectbox(
                            label,
                            values,
                            index=default_index,
                            key=f"pred_{feature}",
                            help=f"Dataset field: {feature}",
                        )
                    else:
                        step = 1.0 if is_integer else 0.1
                        value = int(round(median)) if is_integer else median
                        chosen = st.number_input(
                            label,
                            min_value=int(minimum) if is_integer else minimum,
                            max_value=int(maximum) if is_integer else maximum,
                            value=value,
                            step=int(step) if is_integer else step,
                            key=f"pred_{feature}",
                            help=f"Dataset field: {feature}",
                        )
                        inputs[feature] = int(chosen) if is_integer else float(chosen)

        submitted = st.form_submit_button(
            "🎯 Generate risk estimate",
            type="primary",
            width="stretch",
        )

    if submitted:
        student_df = pd.DataFrame([inputs], columns=X.columns)
        pass_probability = float(prediction_model.predict_proba(student_df)[0, 1])
        risk_probability = 1.0 - pass_probability
        prediction = int(pass_probability >= pass_threshold)

        st.divider()
        left, right = st.columns([1, 1.15])
        with left:
            if prediction == 1:
                st.success("LOWER RISK SIGNAL · Predicted Pass")
            else:
                st.error("AT-RISK SIGNAL · Human review recommended")

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_probability * 100,
                    number={"suffix": "%"},
                    title={"text": "Estimated academic-risk probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "steps": [
                            {"range": [0, 35], "color": "#dcfce7"},
                            {"range": [35, 65], "color": "#fef3c7"},
                            {"range": [65, 100], "color": "#fee2e2"},
                        ],
                        "threshold": {
                            "line": {"color": "#b91c1c", "width": 4},
                            "thickness": 0.75,
                            "value": (1 - pass_threshold) * 100,
                        },
                    },
                )
            )
            gauge.update_layout(height=360, margin={"l": 25, "r": 25, "t": 70, "b": 20})
            st.plotly_chart(gauge, width="stretch")

            c1, c2 = st.columns(2)
            c1.metric("Pass probability", f"{pass_probability:.1%}")
            c2.metric("Risk probability", f"{risk_probability:.1%}")

            if prediction == 0:
                st.markdown(
                    """
                    <div class="callout callout-risk">
                        Recommended action: review attendance, prior failures and support needs with the student. Do not treat this output as proof of ability or intent.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="callout callout-success">
                        A lower-risk prediction does not guarantee success. Continue normal academic monitoring and support.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with right:
            st.markdown("#### Input audit trail")
            display_df = student_df.T.reset_index()
            display_df.columns = ["Dataset feature", "Value"]
            display_df.insert(1, "Readable label", display_df["Dataset feature"].map(feature_label))
            st.dataframe(
                arrow_safe_df(display_df),
                width="stretch",
                hide_index=True,
                height=470,
            )
            prediction_record = student_df.copy()
            prediction_record["model"] = prediction_model_name
            prediction_record["pass_threshold"] = pass_threshold
            prediction_record["pass_probability"] = pass_probability
            prediction_record["risk_probability"] = risk_probability
            prediction_record["prediction"] = "Pass" if prediction == 1 else "At risk"
            st.download_button(
                "📥 Download prediction record",
                data=prediction_record.to_csv(index=False).encode("utf-8"),
                file_name="student_risk_prediction.csv",
                mime="text/csv",
                width="stretch",
            )



# ════════════════════════════════════════
# 🎛️ INTERVENTION SIMULATOR
# ════════════════════════════════════════
elif page == "🎛️ Intervention Simulator":
    st.markdown('<span class="section-kicker">Decision support layer</span>', unsafe_allow_html=True)
    st.title("🎛️ Intervention Simulator")
    st.markdown(
        """
        <div class="signature-strip">
        <strong>What makes this version different:</strong> the app does not stop at a prediction. It shows how an adviser could test realistic support scenarios before deciding how to help a student.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if bundle:
        sim_model_name = bundle["best_model_name"]
        sim_model = bundle["best_pipes"][sim_model_name]
        sim_source = "Best nested-CV model from the current session"
    elif "optuna_models" in st.session_state and st.session_state["optuna_models"]:
        sim_model_name = next(iter(st.session_state["optuna_models"]))
        sim_model = st.session_state["optuna_models"][sim_model_name]
        sim_source = "Optuna-tuned model from the current session"
    else:
        sim_model_name = "Logistic Regression (default)"
        sim_model = get_cached_default_model(X, y)
        sim_source = "Fallback model for demonstration"

    col_a, col_b, col_c = st.columns([1.4, 1, 1])
    with col_a:
        st.info(f"Using: **{sim_model_name}** · {sim_source}")
    with col_b:
        threshold = st.slider("Pass threshold", 0.10, 0.90, 0.50, 0.05, key="sim_threshold")
    with col_c:
        row_idx = st.number_input("Student row index", min_value=0, max_value=len(X)-1, value=0, step=1)

    base_student = X.iloc[[int(row_idx)]].copy()

    def _scenario_row(base: pd.DataFrame, label: str, updates: dict | None = None) -> dict:
        row = base.copy()
        updates = updates or {}
        for key, val in updates.items():
            if key in row.columns and val is not None:
                row.loc[row.index[0], key] = val
        pass_prob = float(sim_model.predict_proba(row)[0, 1])
        risk_prob = 1 - pass_prob
        signal = "Pass" if pass_prob >= threshold else "At risk"
        return {
            "Scenario": label,
            "Pass probability": pass_prob,
            "Risk probability": risk_prob,
            "Signal": signal,
        }

    current_absences = int(base_student["absences"].iloc[0]) if "absences" in base_student.columns else 0
    current_study = int(base_student["studytime"].iloc[0]) if "studytime" in base_student.columns else 1
    current_failures = int(base_student["failures"].iloc[0]) if "failures" in base_student.columns else 0

    st.markdown("### Interactive support controls")
    st.markdown(
        """
        <div class="control-panel">
            <h3>Adviser scenario builder</h3>
            <p>Move the controls below to create a realistic support plan. The interface updates the simulated risk estimate immediately after each change.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        attendance_reduction = st.slider("Attendance improvement", 0, 80, 45, 5, help="Percentage reduction in absences for the simulated support plan.")
    with ctrl2:
        study_boost = st.slider("Study-time boost", 0, 3, 1, 1, help="Increase in studytime level, capped at 4.")
    with ctrl3:
        reduce_failures = st.toggle("Recovery plan", value=True, help="Simulates reducing the failures count by one where possible.")
    with ctrl4:
        enable_support = st.toggle("Activate support services", value=True, help="Sets family/school support fields to yes where those features exist.")

    combined_updates = {
        "absences": max(0, int(current_absences * (1 - attendance_reduction / 100))),
        "studytime": min(4, current_study + int(study_boost)),
    }
    if enable_support and "famsup" in X.columns:
        combined_updates["famsup"] = "yes"
    if enable_support and "schoolsup" in X.columns:
        combined_updates["schoolsup"] = "yes"
    if reduce_failures and "failures" in X.columns:
        combined_updates["failures"] = max(0, current_failures - 1)

    scenarios = [
        _scenario_row(base_student, "Current profile"),
        _scenario_row(base_student, "Attendance support", {"absences": max(0, int(current_absences * 0.65))}),
        _scenario_row(base_student, "Study-time coaching", {"studytime": min(4, current_study + 1)}),
        _scenario_row(base_student, "Custom adviser plan", combined_updates),
    ]
    if "failures" in X.columns:
        scenarios.append(_scenario_row(base_student, "Recovery plan after failures", {"failures": max(0, current_failures - 1), "studytime": min(4, current_study + 1)}))

    scenario_df = pd.DataFrame(scenarios)
    current_risk = float(scenario_df.iloc[0]["Risk probability"])
    preview_cols = st.columns(min(4, len(scenario_df)))
    for col, (_, row) in zip(preview_cols, scenario_df.head(4).iterrows()):
        delta = current_risk - float(row["Risk probability"])
        delta_class = "delta-good" if delta >= 0 else "delta-bad"
        scenario_name = row["Scenario"]
        risk_value = float(row["Risk probability"])
        signal_value = row["Signal"]
        with col:
            st.markdown(
                f"""<div class="scenario-card">
                    <b>{scenario_name}</b><br>
                    <span class="small-note">Risk probability</span>
                    <h3 style="margin:.2rem 0;color:#0f172a;">{risk_value:.1%}</h3>
                    <span class="{delta_class}">{delta:+.1%} vs current</span><br>
                    <span class="small-note">Signal: {signal_value}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    st.subheader("Scenario comparison")
    display_df = scenario_df.copy()
    display_df["Pass probability"] = display_df["Pass probability"].map(lambda v: f"{v:.3f}")
    display_df["Risk probability"] = display_df["Risk probability"].map(lambda v: f"{v:.3f}")
    st.dataframe(arrow_safe_df(display_df), width="stretch", hide_index=True)

    fig = px.bar(
        scenario_df,
        x="Scenario",
        y="Risk probability",
        color="Signal",
        text=scenario_df["Risk probability"].map(lambda v: f"{v:.2f}"),
        title="How support scenarios change estimated risk",
    )
    fig.update_layout(yaxis_range=[0, 1], xaxis_title="", yaxis_title="Estimated risk probability")
    st.plotly_chart(fig, width="stretch")

    best_scenario = scenario_df.sort_values("Risk probability").iloc[0]
    delta = float(scenario_df.iloc[0]["Risk probability"] - best_scenario["Risk probability"])
    if delta > 0.05:
        st.success(
            f"Most promising simulated plan: **{best_scenario['Scenario']}**, reducing estimated risk by approximately **{delta:.1%}** compared with the current profile."
        )
    else:
        st.warning(
            "The scenarios did not substantially reduce predicted risk. This is a signal for human review, not a reason to ignore the student."
        )

    st.markdown("### Adviser notes generated for viva/demo")
    st.markdown(
        f"""
        - Current row: **{int(row_idx)}**  
        - Current absences: **{current_absences}**  
        - Current studytime level: **{current_study}**  
        - Previous failures: **{current_failures}**  
        - The simulator is **counterfactual-support oriented**, not causal proof. It helps demonstrate how an adviser could compare support options before acting.
        """
    )

    st.download_button(
        "📥 Download intervention simulation CSV",
        data=scenario_df.to_csv(index=False).encode("utf-8"),
        file_name="intervention_simulation.csv",
        mime="text/csv",
        width="stretch",
    )

# ════════════════════════════════════════
# 🧭 EVIDENCE CONTROL ROOM
# ════════════════════════════════════════
elif page == "🧭 Evidence Control Room":
    st.markdown('<span class="section-kicker">Distinction-focused project governance</span>', unsafe_allow_html=True)
    st.title("🧭 Evidence Control Room")
    st.caption("A single page that shows the examiner the project is not just an app, but an auditable digital-technology artefact.")

    st.markdown(
        """
        <div class="journey">
            <div class="journey-step"><b>1 · Data audit</b><span>Source, target, leakage control and feature policy.</span></div>
            <div class="journey-step"><b>2 · Model testing</b><span>Baseline, nested CV, risk-focused metrics and statistics.</span></div>
            <div class="journey-step"><b>3 · Responsible AI</b><span>Fairness, calibration, SHAP and human oversight.</span></div>
            <div class="journey-step"><b>4 · Deployment</b><span>Streamlit, GitHub, evidence exports and reproducible artefacts.</span></div>
            <div class="journey-step"><b>5 · Reflection</b><span>Limitations, improvement plan and professional development.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    readiness_rows = [
        {"Area": "Unique visual identity", "Evidence in app": "Custom command-centre layout, branded cards, workflow strip and decision-support pages", "Status": "Strong"},
        {"Area": "Technical depth", "Evidence in app": "Nested CV, baseline, Optuna, risk-focused metrics, threshold-cost analysis", "Status": "Strong"},
        {"Area": "Responsible AI", "Evidence in app": "Fairness diagnostics, calibration, SHAP, model card and usage warnings", "Status": "Strong"},
        {"Area": "Practical value", "Evidence in app": "Intervention Simulator converts predictions into support planning", "Status": "Strong"},
        {"Area": "Reproducibility", "Evidence in app": "Evidence pack, CSV exports, model download, environment metadata", "Status": "Strong"},
        {"Area": "External validity", "Evidence in app": "Clearly states UCI dataset limitations and need for local PDP validation", "Status": "Moderate"},
        {"Area": "Production readiness", "Evidence in app": "Prototype only; no live SIS integration or real student deployment", "Status": "Limited"},
    ]
    readiness_df = pd.DataFrame(readiness_rows)
    st.dataframe(arrow_safe_df(readiness_df), width="stretch", hide_index=True)

    status_counts = readiness_df["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    fig = px.pie(status_counts, names="Status", values="Count", title="Evidence strength overview", hole=.55)
    st.plotly_chart(fig, width="stretch")

    if bundle:
        best = bundle["best_model_name"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best model", best)
        c2.metric("Macro-F1", f"{metric_mean(bundle, best, 'Macro-F1'):.3f}")
        c3.metric("Risk recall", f"{metric_mean(bundle, best, 'Risk Recall'):.3f}")
        c4.metric("ROC-AUC", f"{metric_mean(bundle, best, 'AUC-ROC'):.3f}")
    else:
        st.warning("Train models first to populate live evidence metrics.")

    st.markdown("### What this app can claim")
    st.success("It can claim to be a transparent, research-informed academic prototype for early-warning analysis.")
    st.markdown("### What this app must not claim")
    st.error("It must not claim production readiness, causal proof or direct suitability for PDP University without local validation.")

    viva_cards = pd.DataFrame([
        {"Likely viva question": "What makes your app different from a basic Streamlit dashboard?", "Answer focus": "It includes risk-cost simulation, fairness diagnostics, calibration, SHAP and an auditable evidence pack."},
        {"Likely viva question": "Why should an examiner trust the evaluation?", "Answer focus": "Nested CV separates tuning and testing; baseline and statistical tests reduce overclaiming."},
        {"Likely viva question": "What is the biggest limitation?", "Answer focus": "External validity: the dataset is not PDP University data and must be locally validated."},
    ])
    st.dataframe(arrow_safe_df(viva_cards), width="stretch", hide_index=True)

    st.download_button(
        "📥 Download evidence-control checklist",
        data=readiness_df.to_csv(index=False).encode("utf-8"),
        file_name="distinction_evidence_control_room.csv",
        mime="text/csv",
        width="stretch",
    )

# ℹ️ MODEL CARD
# ════════════════════════════════════════
elif page == "ℹ️ Model Card":
    render_page_header(
        "MODEL GOVERNANCE",
        "Model Card and Reproducibility Record",
        "Document intended use, validation design, limitations, ethical boundaries and the exact experiment environment.",
    )

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(
            f"""
            <div class="panel">
                <h3>Purpose and intended use</h3>
                <p>Estimate the probability that a student will achieve <code>G3 ≥ 10</code> using information available before final grades.</p>
                <p><b>Intended use:</b> academic research, demonstration and human-reviewed early support.</p>
                <p><b>Out-of-scope:</b> automatic sanctions, admissions decisions, causal diagnosis, or direct deployment at PDP University.</p>
                <h3>Dataset boundary</h3>
                <ul>
                    <li>Source: UCI Student Performance dataset (Cortez and Silva, 2008)</li>
                    <li>Not PDP University student data</li>
                    <li>{len(df)} rows; {X.shape[1]} model inputs</li>
                    <li>Target coding: 0 = Fail/at-risk, 1 = Pass</li>
                    <li>G1, G2 and G3 excluded from model inputs</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        metadata = {
            "Dataset fingerprint": dataset_fingerprint(df),
            "Random seed": RANDOM_STATE,
            "Python": platform.python_version(),
            "Streamlit": st.__version__,
            "scikit-learn": sklearn.__version__,
            "Plotly": plotly.__version__,
            "Primary metric": PRIMARY_METRIC,
            "Training status": "Completed" if bundle else "Not run in this session",
            "Training timestamp UTC": bundle.get("trained_at_utc", "—") if bundle else "—",
        }
        st.markdown('<div class="panel"><h3>Reproducibility metadata</h3>', unsafe_allow_html=True)
        st.dataframe(
            arrow_safe_df(pd.DataFrame(metadata.items(), columns=["Field", "Value"])),
            width="stretch",
            hide_index=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Validation design")
    validation = pd.DataFrame(
        [
            {"Component": "Quick experiment", "Implementation": "3 outer folds × 3 inner folds", "Purpose": "Live demonstration and functional verification"},
            {"Component": "Full experiment", "Implementation": "5 outer folds × 5 inner folds", "Purpose": "Stronger final research evidence"},
            {"Component": "Primary selection metric", "Implementation": "Macro-F1", "Purpose": "Avoid majority-class inflation"},
            {"Component": "Risk evidence", "Implementation": "Risk Recall, F1-Risk, missed-risk rate, risk PR-AUC", "Purpose": "Align evaluation with early-warning use"},
            {"Component": "Reliability evidence", "Implementation": "Fold results, bootstrap CIs, Wilcoxon test, calibration", "Purpose": "Quantify uncertainty and stability"},
            {"Component": "Responsible AI", "Implementation": "Subgroup analysis, SHAP and human-review warning", "Purpose": "Expose disparity and black-box risks"},
        ]
    )
    st.dataframe(arrow_safe_df(validation), width="stretch", hide_index=True)

    st.markdown("### Known limitations")
    limitations = pd.DataFrame(
        [
            {"Limitation": "External validity", "Impact": "No evidence that performance transfers to PDP University or another country/institution.", "Required mitigation": "Independent local validation before operational use."},
            {"Limitation": "Sample size", "Impact": "Confidence intervals and subgroup estimates can be unstable.", "Required mitigation": "Use a larger, more recent and representative dataset."},
            {"Limitation": "Construct definition", "Impact": "G3 < 10 captures one form of risk but not wellbeing, dropout or long-term achievement.", "Required mitigation": "Validate the outcome definition with academic stakeholders."},
            {"Limitation": "Causality", "Impact": "SHAP and feature associations do not prove that changing a feature changes the outcome.", "Required mitigation": "Avoid causal recommendations; use human review and additional evidence."},
            {"Limitation": "Fairness", "Impact": "Sensitive-group estimates may reflect historical or sampling bias.", "Required mitigation": "Monitor disparities with sufficient subgroup samples and governance review."},
        ]
    )
    st.dataframe(arrow_safe_df(limitations), width="stretch", hide_index=True)

    if bundle:
        best_name = bundle["best_model_name"]
        claim_title, claim_text = performance_statement(bundle)
        st.markdown("### Current selected model")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", best_name)
        c2.metric("Macro-F1", f"{metric_mean(bundle, best_name, 'Macro-F1'):.3f} ± {metric_std(bundle, best_name, 'Macro-F1'):.3f}")
        c3.metric("Risk Recall", f"{metric_mean(bundle, best_name, 'Risk Recall'):.3f}")
        c4.metric("AUC-ROC", f"{metric_mean(bundle, best_name, 'AUC-ROC'):.3f}")
        st.json(bundle["results"][best_name]["best_params"])
        st.markdown(
            f'<div class="callout"><b>{claim_title}</b><br>{claim_text}</div>',
            unsafe_allow_html=True,
        )
        download_left, download_right = st.columns(2)
        with download_left:
            st.download_button(
                "📄 Download executive summary",
                data=build_executive_summary(bundle, df, X).encode("utf-8"),
                file_name="executive_summary.md",
                mime="text/markdown",
                width="stretch",
            )
        with download_right:
            st.download_button(
                "🗂️ Download auditable evidence pack",
                data=build_evidence_pack(bundle, df, X),
                file_name="btec_ml_evidence_pack.zip",
                mime="application/zip",
                width="stretch",
            )
    else:
        st.info("No completed experiment is available in this browser session.")


# ════════════════════════════════════════
# 📚 RESEARCH EVIDENCE
# ════════════════════════════════════════
elif page == "📚 Research Evidence":
    render_page_header(
        "BTEC LEVEL 6 ALIGNMENT",
        "Research Evidence and Viva Map",
        "Show what the application demonstrates and where the written report must provide the remaining critical argument.",
    )

    st.markdown(
        """
        <div class="callout">
            <b>No application can guarantee a Distinction.</b> The app supplies technical evidence mainly for implementation, analysis, professional communication and validity/reliability. D1, D2 and D4 still require critical written evaluation in the report and viva.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Project aim and research questions")
    st.markdown(
        """
        **Aim.** To develop and critically evaluate an explainable machine-learning early-warning prototype for identifying students at risk of failing, using the UCI Student Performance dataset.

        **RQ1.** Which candidate classifier provides the strongest balanced and risk-focused out-of-fold performance?

        **RQ2.** How stable and well calibrated are the selected model's predictions under nested cross-validation?

        **RQ3.** Do risk-detection performance and error rates vary across selected student subgroups?

        **RQ4.** Which features most strongly influence global and individual model predictions, and what limitations constrain their interpretation?
        """
    )

    st.markdown("### Application evidence mapped to assessment criteria")
    mapping = pd.DataFrame(
        [
            {"Criterion": "P4", "Application evidence": "Deployed end-to-end pipeline, trained models, prediction interface and exported artefacts.", "Strength": "Strong", "Report still needed": "Explain deviations from the original plan."},
            {"Criterion": "P5", "Application evidence": "Secondary dataset processing, EDA, nested model training and generated findings.", "Strength": "Strong", "Report still needed": "Document collection source, preprocessing and sampling limitations."},
            {"Criterion": "M3", "Application evidence": "Methods aligned to objectives: Macro-F1 tuning, risk metrics, calibration, fairness and SHAP.", "Strength": "Strong", "Report still needed": "Cite and justify every analytical choice."},
            {"Criterion": "M4", "Application evidence": "Leaderboard, fold trends, ROC/PR, confusion matrix, cost-sensitive threshold analysis and subgroup visualisation.", "Strength": "Strong", "Report still needed": "Write reasoned cross-comparisons and conclusions."},
            {"Criterion": "D3", "Application evidence": "Nested CV, confidence intervals, calibration, corrected statistical comparisons, learning curves, error analysis and external-validity warnings.", "Strength": "Strong technical evidence", "Report still needed": "Critically evaluate validity, reliability and implications."},
            {"Criterion": "P6/M5", "Application evidence": "Professional interactive dashboard, downloadable evidence pack and responsible-use communication.", "Strength": "Strong", "Report still needed": "Structured report, Harvard citations and audience-tailored viva."},
            {"Criterion": "D1/D2/D4", "Application evidence": "Only limited support through alternative-metric rationale and limitations.", "Strength": "Insufficient alone", "Report still needed": "Alternative research directions, project-management critique and reflective development."},
        ]
    )
    st.dataframe(arrow_safe_df(mapping), width="stretch", hide_index=True)

    st.markdown("### Defensible viva statements")
    viva = pd.DataFrame(
        [
            {"Question": "Why not use ordinary accuracy or pass-class F1?", "Defensible answer": "The Pass class is larger, so those metrics can reward a model that misses at-risk students. Macro-F1 and risk recall align evaluation with the early-warning problem."},
            {"Question": "How did you reduce data leakage?", "Defensible answer": "Imputation, scaling and encoding are inside a pipeline fitted separately within each cross-validation training fold."},
            {"Question": "Why nested cross-validation?", "Defensible answer": "The inner loop selects hyperparameters while the untouched outer loop estimates generalisation, reducing optimistic model-selection bias."},
            {"Question": "Can the model be used at PDP University?", "Defensible answer": "No direct deployment claim is justified because the dataset is not from PDP and there is no external local validation."},
            {"Question": "Does SHAP prove causes?", "Defensible answer": "No. SHAP explains how the fitted model distributes prediction contributions; it does not establish causal effects."},
            {"Question": "What is the most serious error?", "Defensible answer": "An actual at-risk student predicted as Pass. The app reports this explicitly as a missed-risk case and allows threshold analysis."},
        ]
    )
    st.dataframe(arrow_safe_df(viva), width="stretch", hide_index=True)

    st.markdown("### Final app acceptance checklist")
    checklist = pd.DataFrame(
        [
            {"Check": "Quick experiment completes without Traceback", "Required": "Yes"},
            {"Check": "Full 5×5 experiment exported at least once", "Required": "Recommended for final evidence"},
            {"Check": "Best model exceeds Dummy baseline on Macro-F1", "Required": "Yes, or critically explain failure"},
            {"Check": "Missed-risk and false-alarm errors are reported correctly", "Required": "Yes"},
            {"Check": "Fairness page tested for at least two subgroup features", "Required": "Yes"},
            {"Check": "SHAP global and local views load successfully", "Required": "Yes"},
            {"Check": "Evidence ZIP downloads and opens", "Required": "Yes"},
            {"Check": "No claim of production readiness or causal prediction", "Required": "Yes"},
        ]
    )
    st.dataframe(arrow_safe_df(checklist), width="stretch", hide_index=True)

    st.markdown("### Academic foundations used by the application")
    references = pd.DataFrame(
        [
            {
                "Method": "Dataset and prediction problem",
                "Harvard-style source": "Cortez, P. and Silva, A.M.G. (2008) Using data mining to predict secondary school student performance.",
            },
            {
                "Method": "Nested model selection",
                "Harvard-style source": "Varma, S. and Simon, R. (2006) 'Bias in error estimation when using cross-validation for model selection', BMC Bioinformatics, 7, 91.",
            },
            {
                "Method": "Machine-learning implementation",
                "Harvard-style source": "Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', Journal of Machine Learning Research, 12, pp. 2825–2830.",
            },
            {
                "Method": "Explainable AI",
                "Harvard-style source": "Lundberg, S.M. and Lee, S.-I. (2017) 'A Unified Approach to Interpreting Model Predictions', Advances in Neural Information Processing Systems, 30.",
            },
            {
                "Method": "Model governance",
                "Harvard-style source": "Mitchell, M. et al. (2019) 'Model Cards for Model Reporting', Proceedings of the Conference on Fairness, Accountability, and Transparency.",
            },
        ]
    )
    st.dataframe(arrow_safe_df(references), width="stretch", hide_index=True)
