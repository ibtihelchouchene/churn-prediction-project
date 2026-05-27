"""utils/charts.py — Reusable Plotly chart builders."""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ── Shared theme ───────────────────────────────────────────────────────────────
THEME = dict(
    paper_bgcolor="#0F2035",
    plot_bgcolor="#0F2035",
    font=dict(family="DM Sans, sans-serif", color="#B0BEC5", size=12),
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(gridcolor="#1E2D3D", zerolinecolor="#1E2D3D"),
    yaxis=dict(gridcolor="#1E2D3D", zerolinecolor="#1E2D3D"),
)

COLORS = {
    "accent":  "#00E5FF",
    "accent2": "#FF6B35",
    "accent3": "#A78BFA",
    "high":    "#FF3D57",
    "med":     "#FFB300",
    "low":     "#00C853",
    "muted":   "#546E7A",
}

RISK_COLORS = {"High Risk": COLORS["high"], "Medium Risk": COLORS["med"], "Low Risk": COLORS["low"]}


def _apply_theme(fig):
    fig.update_layout(**THEME)
    return fig


# ── ROC Curve ─────────────────────────────────────────────────────────────────
def roc_curve_fig(fpr, tpr, auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name=f"Model (AUC = {auc:.3f})",
        line=dict(color=COLORS["accent"], width=2.5),
        fill="tozeroy", fillcolor="rgba(0,229,255,0.06)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color=COLORS["muted"], width=1, dash="dash"),
    ))
    fig.update_layout(
        title=dict(text="ROC Curve", font=dict(family="Space Mono, monospace", size=13)),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        **THEME,
    )
    return fig


# ── PR Curve ──────────────────────────────────────────────────────────────────
def pr_curve_fig(prec, rec, ap, baseline):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rec, y=prec, mode="lines", name=f"Model (AP = {ap:.3f})",
        line=dict(color=COLORS["accent3"], width=2.5),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.06)",
    ))
    fig.add_hline(y=baseline, line_dash="dash", line_color=COLORS["muted"],
                  annotation_text=f"Baseline ({baseline:.2f})")
    fig.update_layout(
        title=dict(text="Precision-Recall Curve", font=dict(family="Space Mono, monospace", size=13)),
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        **THEME,
    )
    return fig


# ── Confusion Matrix ──────────────────────────────────────────────────────────
def confusion_matrix_fig(tn, fp, fn, tp):
    z    = [[tn, fp], [fn, tp]]
    text = [[f"TN\n{tn:,}", f"FP\n{fp:,}"], [f"FN\n{fn:,}", f"TP\n{tp:,}"]]
    fig  = go.Figure(go.Heatmap(
        z=z, text=text, texttemplate="%{text}",
        x=["Pred: No Churn", "Pred: Churn"],
        y=["Act: No Churn",  "Act: Churn"],
        colorscale=[[0, "#0A1628"], [1, "#00E5FF"]],
        showscale=False,
        textfont=dict(size=14, family="Space Mono, monospace"),
    ))
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fig.update_layout(
        title=dict(
            text=f"Confusion Matrix  |  Recall: {recall:.1%}  Precision: {precision:.1%}",
            font=dict(family="Space Mono, monospace", size=12),
        ),
        **THEME,
    )
    return fig


# ── Feature Importance ────────────────────────────────────────────────────────
def feature_importance_fig(df, title="Feature Importance"):
    df = df.sort_values("importance" if "importance" in df.columns else "mean_shap")
    val_col = "importance" if "importance" in df.columns else "mean_shap"
    fig = go.Figure(go.Bar(
        x=df[val_col], y=df["feature"],
        orientation="h",
        marker=dict(
            color=df[val_col],
            colorscale=[[0, "#1E2D3D"], [1, "#00E5FF"]],
            showscale=False,
        ),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Mono, monospace", size=13)),
        xaxis_title="Importance",
        height=max(380, len(df) * 22),
        **THEME,
    )
    return fig


# ── Risk donut ────────────────────────────────────────────────────────────────
def risk_donut_fig(risk_counts):
    labels = list(risk_counts.keys())
    values = list(risk_counts.values())
    colors = [RISK_COLORS.get(l, "#546E7A") for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#0A1628", width=3)),
        textinfo="label+percent",
        textfont=dict(size=11),
    ))
    total = sum(values)
    fig.update_layout(
        title=dict(text="Risk Segmentation", font=dict(family="Space Mono, monospace", size=13)),
        annotations=[dict(text=f"{total:,}<br>customers", x=0.5, y=0.5,
                          font=dict(size=14, color="#E0E6ED"), showarrow=False)],
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        **THEME,
    )
    return fig


# ── Probability histogram ─────────────────────────────────────────────────────
def prob_histogram_fig(probs, actuals=None):
    fig = go.Figure()
    if actuals is not None:
        for label, color, name in [(0, COLORS["low"], "No Churn"), (1, COLORS["high"], "Churn")]:
            mask = actuals == label
            fig.add_trace(go.Histogram(
                x=probs[mask], name=name, nbinsx=40,
                marker_color=color, opacity=0.75,
            ))
        fig.update_layout(barmode="overlay")
    else:
        fig.add_trace(go.Histogram(x=probs, nbinsx=40, marker_color=COLORS["accent"], opacity=0.8))
    fig.update_layout(
        title=dict(text="Churn Probability Distribution", font=dict(family="Space Mono, monospace", size=13)),
        xaxis_title="Churn Probability",
        yaxis_title="Count",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        **THEME,
    )
    return fig


# ── Risk bar chart ────────────────────────────────────────────────────────────
def risk_bar_fig(risk_counts):
    labels = list(risk_counts.keys())
    values = list(risk_counts.values())
    colors = [RISK_COLORS.get(l, "#546E7A") for l in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=values, textposition="outside",
        textfont=dict(family="Space Mono, monospace", size=11),
    ))
    fig.update_layout(
        title=dict(text="Customers by Risk Level", font=dict(family="Space Mono, monospace", size=13)),
        yaxis_title="Count",
        showlegend=False,
        **THEME,
    )
    return fig


# ── SHAP local waterfall ──────────────────────────────────────────────────────
def shap_local_fig(shap_df):
    if shap_df is None or shap_df.empty:
        return go.Figure().update_layout(
            title="SHAP not available", **THEME
        )
    shap_df = shap_df.sort_values("shap_value")
    colors  = [COLORS["high"] if v > 0 else COLORS["low"] for v in shap_df["shap_value"]]
    fig = go.Figure(go.Bar(
        x=shap_df["shap_value"],
        y=shap_df["feature"],
        orientation="h",
        marker_color=colors,
    ))
    fig.add_vline(x=0, line_color="#546E7A", line_width=1)
    fig.update_layout(
        title=dict(text="SHAP — Feature Contributions", font=dict(family="Space Mono, monospace", size=12)),
        xaxis_title="SHAP Value  (→ increases churn risk)",
        height=400,
        **THEME,
    )
    return fig


# ── Metric gauge card (returns dict for display) ──────────────────────────────
def metric_gauge_fig(value, title, max_val=1.0):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title=dict(text=title, font=dict(family="Space Mono, monospace", size=11)),
        number=dict(suffix="%", font=dict(family="Space Mono, monospace", size=24, color="#00E5FF")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#546E7A"),
            bar=dict(color="#00E5FF"),
            bgcolor="#0A1628",
            bordercolor="#1E2D3D",
            steps=[
                dict(range=[0, 50],  color="#0F1E2E"),
                dict(range=[50, 75], color="#0F2035"),
                dict(range=[75, 100],color="#162840"),
            ],
        ),
    ))
    fig.update_layout(paper_bgcolor="#0F2035", font=dict(color="#B0BEC5"), height=200, margin=dict(l=20, r=20, t=40, b=10))
    return fig
