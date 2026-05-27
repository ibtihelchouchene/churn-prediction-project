# dashboard_v2.py — Telecom Churn Dashboard
# Reuses ALL outputs from the existing pipeline scripts.
# No model training. No preprocessing. No EDA recomputation.

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, warnings, pickle
import numpy as np, pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    brier_score_loss, precision_recall_curve,
    roc_curve, confusion_matrix,
)
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(BASE_DIR, "Telecom_customer churn.csv")
LGB_PATH = os.path.join(DATA_DIR, "best_lgb.pkl")
XGB_PATH = os.path.join(DATA_DIR, "best_xgb.pkl")
CV_PATH  = os.path.join(DATA_DIR, "cv_results.json")
TN_PATH  = os.path.join(DATA_DIR, "tuning_summary.json")

# ══════════════════════════════════════════════════════════════════════════════
# PALETTE & BASE LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
C_BG    = "#0D1B2A"
C_CARD  = "#0F2035"
C_DEEP  = "#0A1628"
C_BORD  = "#1E2D3D"
C_ACC   = "#00E5FF"
C_ACC2  = "#FF6B35"
C_ACC3  = "#A78BFA"
C_GREEN = "#00C853"
C_RED   = "#FF3D57"
C_AMBER = "#FFB300"
C_MUTED = "#546E7A"
PALETTE = [C_ACC, C_ACC2, C_ACC3, C_GREEN, C_RED, C_AMBER, "#38bdf8", "#e879f9"]

BL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#B0BEC5", family="DM Sans, sans-serif"),
    margin=dict(l=45, r=20, t=45, b=40), colorway=PALETTE,
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_BORD),
    xaxis=dict(gridcolor=C_BORD, zerolinecolor=C_BORD),
    yaxis=dict(gridcolor=C_BORD, zerolinecolor=C_BORD),
)
G  = {"displayModeBar": False}
TS = {"backgroundColor": C_CARD,  "color": "#94a3b8", "border": f"1px solid {C_BORD}"}
TA = {"backgroundColor": C_BORD,  "color": "#e2e8f0", "border": f"1px solid {C_ACC}"}
DD = {"backgroundColor": C_CARD,  "color": "#e2e8f0"}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD MODEL 
# ══════════════════════════════════════════════════════════════════════════════
print("Loading pipeline outputs…")
MODEL, MODEL_NAME = None, "Unknown"
for path, name in [(LGB_PATH, "LightGBM"), (XGB_PATH, "XGBoost")]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            MODEL = pickle.load(f)
        MODEL_NAME = name
        print(f"  [OK] Model: {name}")
        break
if MODEL is None:
    raise FileNotFoundError("No model pkl in data/. Run 04_tuning.py first.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — LOAD PREPROCESSED SPLITS 
# ══════════════════════════════════════════════════════════════════════════════
X_TRAIN = pd.read_parquet(os.path.join(DATA_DIR, "X_train.parquet")).astype(np.float32)
X_TEST  = pd.read_parquet(os.path.join(DATA_DIR, "X_test.parquet")).astype(np.float32)
Y_TRAIN = pd.read_parquet(os.path.join(DATA_DIR, "y_train.parquet")).squeeze()
Y_TEST  = pd.read_parquet(os.path.join(DATA_DIR, "y_test.parquet")).squeeze()
print(f"  [OK] Splits: train {X_TRAIN.shape}, test {X_TEST.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — LOAD RAW CSV
# ══════════════════════════════════════════════════════════════════════════════
RAW_DF     = pd.read_csv(CSV_PATH)
N_TOTAL    = len(RAW_DF)
CHURN_RATE = float(RAW_DF["churn"].mean())
N_CHURN    = int(RAW_DF["churn"].sum())
N_NO       = N_TOTAL - N_CHURN
print(f"  [OK] Raw CSV: {N_TOTAL:,} rows")

DROP_EDA = {
    "datovr_Mean","recv_sms_Mean","unan_dat_Mean","callfwdv_Mean","mou_pead_Mean",
    "drop_dat_Mean","totmou","totcalls","mou_opkd_Mean","avg6qty","drop_blk_Mean",
    "attempt_Mean","complete_Mean","ccrndmou_Mean","mouowylisv_Mean","mouiwylisv_Mean",
    "mou_rvce_Mean","threeway_Mean","da_Mean","adjmou","adjqty","blck_dat_Mean",
    "totrev","adjrev","truck","rv","income","numbcars","forgntvl","Customer_ID",
    "new_cell","dwllsize","kid3_5","kid6_10","kid11_15","kid16_17","churn",
}
NUM_COLS = sorted([c for c in RAW_DF.select_dtypes(include=[np.number]).columns if c not in DROP_EDA])
CAT_COLS = [c for c in ["crclscod","asl_flag","prizm_social_one","area","dualband",
    "refurb_new","hnd_webcap","ownrent","dwlltype","marital","infobase",
    "HHstatin","ethnic","creditcd","kid0_2"] if c in RAW_DF.columns]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — LOAD CV & TUNING RESULTS
# ══════════════════════════════════════════════════════════════════════════════
CV_DATA = {}
if os.path.exists(CV_PATH):
    CV_DATA = pd.read_json(CV_PATH).to_dict()
    print(f"  [OK] CV results: {list(CV_DATA.keys())}")

TUNE_DATA = {}
if os.path.exists(TN_PATH):
    TUNE_DATA = pd.read_json(TN_PATH, orient="records").set_index("Model").to_dict("index")
    print(f"  [OK] Tuning summary: {list(TUNE_DATA.keys())}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — PREDICTIONS & THRESHOLD 
# ══════════════════════════════════════════════════════════════════════════════
PROB_TRAIN = MODEL.predict_proba(X_TRAIN)[:, 1]
PROB_TEST  = MODEL.predict_proba(X_TEST)[:, 1]
prec_, rec_, thr_ = precision_recall_curve(Y_TRAIN, PROB_TRAIN)
f1s_ = np.where((prec_ + rec_) == 0, 0, 2*prec_*rec_/(prec_+rec_))
best_i  = int(np.argmax(f1s_))
OPT_THR = float(thr_[best_i-1]) if best_i > 0 else 0.5
PRED_TEST = (PROB_TEST >= OPT_THR).astype(int)
print(f"  [OK] Predictions done. Optimal threshold: {OPT_THR:.3f}")

def risk_label(p):
    if p >= 0.70: return "High Risk"
    if p >= 0.40: return "Medium Risk"
    return "Low Risk"
RISK = np.array([risk_label(p) for p in PROB_TEST])

RISK_COUNTS = {
    "High Risk":   int((RISK == "High Risk").sum()),
    "Medium Risk": int((RISK == "Medium Risk").sum()),
    "Low Risk":    int((RISK == "Low Risk").sum()),
}

FEATURES = X_TEST.columns.tolist()
FI_DF = pd.DataFrame({
    "feature":    FEATURES,
    "importance": MODEL.feature_importances_ if hasattr(MODEL, "feature_importances_") else np.ones(len(FEATURES)),
}).sort_values("importance", ascending=False).head(20)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — METRICS HELPER 
# ══════════════════════════════════════════════════════════════════════════════
def get_metrics(threshold):
    yp = (PROB_TEST >= threshold).astype(int)
    return {
        "Accuracy":  accuracy_score(Y_TEST, yp),
        "Precision": precision_score(Y_TEST, yp, zero_division=0),
        "Recall":    recall_score(Y_TEST, yp, zero_division=0),
        "F1":        f1_score(Y_TEST, yp, zero_division=0),
        "ROC-AUC":   roc_auc_score(Y_TEST, PROB_TEST),
        "PR-AUC":    average_precision_score(Y_TEST, PROB_TEST),
        "Brier":     brier_score_loss(Y_TEST, PROB_TEST),
    }

M = get_metrics(OPT_THR)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — UI HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def card(title, *children, accent=C_ACC):
    return dbc.Card([
        dbc.CardHeader(title, style={
            "background": C_BORD, "color": "#e2e8f0", "fontWeight": 600,
            "borderRadius": "8px 8px 0 0", "borderLeft": f"3px solid {accent}",
            "fontFamily": "Space Mono, monospace", "fontSize": "11px", "letterSpacing": "1px",
        }),
        dbc.CardBody(list(children)),
    ], style={"background": C_CARD, "border": f"1px solid {C_BORD}",
              "borderRadius": 8, "marginBottom": 20})

def kpi(label, value, sub="", color=C_ACC):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.P(label, className="text-muted mb-1",
               style={"fontSize": "0.72rem", "letterSpacing": "2px",
                      "fontFamily": "Space Mono, monospace"}),
        html.H3(value, style={"color": color, "fontWeight": 700,
                               "fontFamily": "Space Mono, monospace", "fontSize": "1.4rem"}),
        html.P(sub, className="text-muted mb-0", style={"fontSize": "0.72rem"}),
    ]), style={"background": C_CARD, "border": f"1px solid {C_BORD}",
               "borderRadius": 8, "borderTop": f"2px solid {color}"}),
    md=2, className="mb-3")

def build_cv_table():
    if not CV_DATA:
        return html.P("No CV data found — run 03_baseline_cv.py first.",
                      style={"color": C_MUTED, "fontStyle": "italic"})
    
    # 1. Create the Header
    rows = [html.Thead(html.Tr([html.Th(c) for c in ["Model", "ROC-AUC", "PR-AUC", "F1"]]))]
    
    # Helper function to safely extract the number from the nested dictionary
    def get_numeric_score(metric_key):
        metric_entry = CV_DATA.get(metric_key, 0)
        
        if isinstance(metric_entry, dict):
            # Tries to get 'mean' first; if not found, grabs the first value in the dict
            return metric_entry.get('mean', next(iter(metric_entry.values()), 0))
            
        return metric_entry # Fallback if it's already a float/int

    # 2. Create the row using the helper function
    rows += [
        html.Tr([
            html.Td(MODEL_NAME), 
            html.Td(f"{get_numeric_score('ROC-AUC'):.4f}"),
            html.Td(f"{get_numeric_score('PR-AUC'):.4f}"),
            html.Td(f"{get_numeric_score('F1'):.4f}"),
        ])
    ]
    
    return dbc.Table(rows, bordered=True, striped=True, size="sm", responsive=True)

TABLE_COND = [
    {"if": {"filter_query": '{Risk Level} = "High Risk"',   "column_id": "Risk Level"},
     "color": C_RED, "fontWeight": "600"},
    {"if": {"filter_query": '{Risk Level} = "Medium Risk"', "column_id": "Risk Level"},
     "color": C_AMBER, "fontWeight": "600"},
    {"if": {"filter_query": '{Risk Level} = "Low Risk"',    "column_id": "Risk Level"},
     "color": C_GREEN, "fontWeight": "600"},
    {"if": {"filter_query": '{Predicted} = "Churn"', "column_id": "Predicted"},
     "color": C_RED},
    {"if": {"state": "selected"},
     "backgroundColor": "#162840", "border": f"1px solid {C_ACC}"},
    {"if": {"state": "active"}, "backgroundColor": "#162840"},
]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — PRE-BUILT FIGURES 
# ══════════════════════════════════════════════════════════════════════════════
fig_donut = go.Figure(go.Pie(
    labels=["No Churn", "Churn"], values=[N_NO, N_CHURN],
    hole=0.65, marker_colors=[C_GREEN, C_RED], textinfo="percent+label",
    textfont=dict(size=12),
))
fig_donut.update_layout(**BL, title="Churn Distribution", height=300)

fig_risk = go.Figure(go.Bar(
    x=list(RISK_COUNTS.keys()), y=list(RISK_COUNTS.values()),
    marker_color=[C_RED, C_AMBER, C_GREEN],
    text=list(RISK_COUNTS.values()), textposition="outside",
))
fig_risk.update_layout(**BL, title="Customers by Risk Segment",
    yaxis_title="Count", showlegend=False, height=300)

fig_prob_hist = go.Figure()
for label, color, mask in [
    ("No Churn", C_GREEN, Y_TEST.values == 0),
    ("Churn",    C_RED,   Y_TEST.values == 1),
]:
    fig_prob_hist.add_trace(go.Histogram(
        x=PROB_TEST[mask], name=label, nbinsx=40,
        marker_color=color, opacity=0.75,
    ))
fig_prob_hist.update_layout(**BL, barmode="overlay",
    title="Churn Probability Distribution by True Label",
    xaxis_title="Predicted Probability", yaxis_title="Count", height=300)

top20 = RAW_DF[NUM_COLS].var().nlargest(20).index.tolist()
corr_m = RAW_DF[top20].corr()
fig_corr = go.Figure(go.Heatmap(
    z=corr_m.values, x=corr_m.columns.tolist(), y=corr_m.index.tolist(),
    colorscale="RdBu_r", zmid=0,
    hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
))
fig_corr.update_layout(**BL,
    title="Correlation Heatmap — Top 20 Numeric Features by Variance", height=520)

def cv_bar(metric_key):
    # 1. Default fallback if the metric doesn't exist
    metric_entry = CV_DATA.get(metric_key, {})
    
    # 2. Extract the list of scores for the folds
    folds_scores = [0, 0, 0] # Default empty bars
    
    if isinstance(metric_entry, dict):
        # Look for common keys where folds might be saved (e.g., 'folds', 'scores', 'values')
        for key in ['folds', 'scores', 'values', 'test_scores']:
            if key in metric_entry and isinstance(metric_entry[key], list):
                folds_scores = metric_entry[key]
                break
        else:
            # If it's a dict but has no list inside, maybe it has keys like 'fold_1', 'fold_2'...
            # Let's extract any numeric values that aren't the 'mean' or 'std'
            numeric_vals = [v for k, v in metric_entry.items() if k not in ['mean', 'std'] and isinstance(v, (int, float))]
            if len(numeric_vals) >= 3:
                folds_scores = numeric_vals[:3]
    elif isinstance(metric_entry, list):
        # If the entry itself is just the list of fold scores
        folds_scores = metric_entry

    # Ensure we only take up to 3 folds for the x-axis alignment
    folds_scores = (folds_scores + [0, 0, 0])[:3]

    # 3. Define clear X-axis labels for the folds
    x_labels = ["Fold 1", "Fold 2", "Fold 3"]

    # 4. Generate the bar figure using Plotly Express
    import plotly.express as px
    fig = px.bar(
        x=x_labels, 
        y=folds_scores,
        text=[f"{v:.4f}" for v in folds_scores],
        labels={'x': 'Cross-Validation Folds', 'y': 'Score'}
    )
    
    # Update text position to sit nicely on top or inside the bars
    fig.update_traces(textposition='auto', marker_color=C_ACC3) # update color to match your UI accent

    # 5. Apply your base layout and individual constraints safely (as fixed earlier!)
    fig.update_layout(**BL)
    fig.update_layout(
        title=f"3-Fold CV - {metric_key}",
        showlegend=False, 
        yaxis=dict(range=[0, 1], gridcolor=C_BORD), 
        xaxis=dict(gridcolor=C_BORD),
        height=340
    )

    return fig

fig_cv_roc = cv_bar("ROC-AUC")
fig_cv_pr  = cv_bar("PR-AUC")
fig_cv_f1  = cv_bar("F1")

if TUNE_DATA:
    tune_df = pd.DataFrame([
        {"Model": k, "CV ROC-AUC (tuned)": v.get("CV ROC-AUC (tuned)", 0)}
        for k, v in TUNE_DATA.items()
    ])
    fig_tune = px.bar(tune_df, x="Model", y="CV ROC-AUC (tuned)",
        color="Model", color_discrete_sequence=PALETTE,
        title="Tuned Model CV ROC-AUC (from 04_tuning.py)")
    fig_tune.update_layout(**BL, showlegend=False, height=300)
else:
    fig_tune = go.Figure().update_layout(**BL,
        title="Tuning results — run 04_tuning.py to populate", height=300)

fpr_t, tpr_t, _ = roc_curve(Y_TEST, PROB_TEST)
auc_val = roc_auc_score(Y_TEST, PROB_TEST)
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
    line=dict(dash="dash", color=C_MUTED), name="Random"))
fig_roc.add_trace(go.Scatter(x=fpr_t, y=tpr_t, mode="lines",
    name=f"{MODEL_NAME} (AUC={auc_val:.4f})",
    line=dict(color=C_ACC, width=2.5),
    fill="tozeroy", fillcolor="rgba(0,229,255,0.06)"))
fig_roc.update_layout(**BL, title="ROC Curve",
    xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=400)

prec_c, rec_c, _ = precision_recall_curve(Y_TEST, PROB_TEST)
ap_val = average_precision_score(Y_TEST, PROB_TEST)
fig_pr = go.Figure()
fig_pr.add_hline(y=CHURN_RATE, line_dash="dash", line_color=C_MUTED,
    annotation_text=f"Baseline ({CHURN_RATE:.2f})")
fig_pr.add_trace(go.Scatter(x=rec_c, y=prec_c, mode="lines",
    name=f"{MODEL_NAME} (AP={ap_val:.4f})",
    line=dict(color=C_ACC3, width=2.5),
    fill="tozeroy", fillcolor="rgba(167,139,250,0.06)"))
fig_pr.update_layout(**BL, title="Precision-Recall Curve",
    xaxis_title="Recall", yaxis_title="Precision", height=400)

fig_fi = go.Figure(go.Bar(
    x=FI_DF["importance"].values[::-1],
    y=FI_DF["feature"].values[::-1],
    orientation="h",
    marker=dict(
        color=FI_DF["importance"].values[::-1],
        colorscale=[[0, C_BORD], [1, C_ACC]], showscale=False,
    ),
))
fig_fi.update_layout(**BL,
    title=f"Top 20 Feature Importances — {MODEL_NAME}", height=560)

KEY_FEATS = ["rev_Mean","mou_Mean","months","eqpdays","change_mou",
             "change_rev","custcare_Mean","ovrmou_Mean","roam_Mean","actvsubs"]
EXPLORER_DF = pd.DataFrame()
EXPLORER_DF["Customer #"] = [f"CUST-{i+1:05d}" for i in range(len(PROB_TEST))]
EXPLORER_DF["Churn Prob"] = (PROB_TEST * 100).round(1)
EXPLORER_DF["Predicted"]  = ["Churn" if p else "No Churn" for p in PRED_TEST]
EXPLORER_DF["Risk Level"] = RISK
for f in KEY_FEATS:
    if f in X_TEST.columns:
        EXPLORER_DF[f] = X_TEST[f].values.round(2)

print("  [OK] All figures built -- starting dashboard\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap",
    ],
    title="Churn Intelligence",
    suppress_callback_exceptions=True,
)
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div("TELECOM", style={"fontFamily": "Space Mono, monospace",
                "fontSize": "10px", "letterSpacing": "6px", "color": C_MUTED}),
            html.H1("Churn Intelligence Dashboard", style={
                "color": C_ACC, "fontWeight": 700,
                "fontFamily": "Space Mono, monospace", "fontSize": "26px",
            }),
            html.P(
                f"Model: {MODEL_NAME}  ·  {N_TOTAL:,} customers  ·  "
                f"Optimal threshold: {OPT_THR:.3f}  ·  ROC-AUC: {M['ROC-AUC']:.4f}",
                style={"color": C_MUTED, "fontSize": "13px"},
            ),
        ], width=9),
        dbc.Col(html.Div([
            dbc.Badge(f"{N_TOTAL:,} rows",        color="info",      className="me-2"),
            dbc.Badge(MODEL_NAME,                  color="success",   className="me-2"),
            dbc.Badge(f"AUC {M['ROC-AUC']:.4f}",  color="primary"),
        ], className="d-flex align-items-center justify-content-end h-100"), width=3),
    ], className="mb-3 mt-4"),

    dbc.Row([
        kpi("TOTAL CUSTOMERS",  f"{N_TOTAL:,}",           f"{X_TEST.shape[1]} features"),
        kpi("CHURN RATE",       f"{CHURN_RATE:.1%}",      f"{N_CHURN:,} churned",        color=C_RED),
        kpi("PRED CHURNERS",    f"{PRED_TEST.sum():,}",   f"thr = {OPT_THR:.2f}",        color=C_AMBER),
        kpi("AVG CHURN PROB",   f"{PROB_TEST.mean():.3f}","test set mean"),
        kpi("HIGH RISK",        f"{RISK_COUNTS['High Risk']:,}", "prob ≥ 0.70",          color=C_ACC2),
        kpi("ROC-AUC",          f"{M['ROC-AUC']:.4f}",   f"F1 = {M['F1']:.3f}",         color=C_ACC3),
    ]),

    dcc.Tabs(value="t-overview", style={"marginBottom": 20}, children=[
        dcc.Tab(label="Overview", value="t-overview", style=TS, selected_style=TA, children=[
            dbc.Row([
                dbc.Col(card("CHURN DISTRIBUTION", dcc.Graph(figure=fig_donut, config=G)), md=4),
                dbc.Col(card("RISK SEGMENTATION", dcc.Graph(figure=fig_risk, config=G)), md=4),
                dbc.Col(card("PROBABILITY DISTRIBUTION", dcc.Graph(figure=fig_prob_hist, config=G)), md=4),
            ]),
            dbc.Row([
                dbc.Col(card("DATASET SUMMARY", dbc.Table([
                    html.Thead(html.Tr([html.Th("Statistic"), html.Th("Value")])),
                    html.Tbody([
                        html.Tr([html.Td("Total records"),         html.Td(f"{N_TOTAL:,}")]),
                        html.Tr([html.Td("Churned"),               html.Td(f"{N_CHURN:,}  ({CHURN_RATE:.1%})")]),
                        html.Tr([html.Td("Not Churned"),           html.Td(f"{N_NO:,}  ({1-CHURN_RATE:.1%})")]),
                        html.Tr([html.Td("Train size"),            html.Td(f"{len(X_TRAIN):,}")]),
                        html.Tr([html.Td("Test size"),             html.Td(f"{len(X_TEST):,}")]),
                        html.Tr([html.Td("Features (after VIF)"),  html.Td(f"{X_TEST.shape[1]}")]),
                        html.Tr([html.Td("Best model"),            html.Td(MODEL_NAME)]),
                        html.Tr([html.Td("Optimal threshold"),     html.Td(f"{OPT_THR:.4f}")]),
                        html.Tr([html.Td("ROC-AUC"),               html.Td(f"{M['ROC-AUC']:.4f}")]),
                        html.Tr([html.Td("PR-AUC"),                html.Td(f"{M['PR-AUC']:.4f}")]),
                        html.Tr([html.Td("F1"),                    html.Td(f"{M['F1']:.4f}")]),
                        html.Tr([html.Td("Recall"),                html.Td(f"{M['Recall']:.4f}")]),
                        html.Tr([html.Td("Precision"),             html.Td(f"{M['Precision']:.4f}")]),
                    ])
                ], bordered=True, striped=True, size="sm")), md=4),
                dbc.Col(card("CORRELATION HEATMAP — TOP 20 FEATURES BY VARIANCE",
                    dcc.Graph(figure=fig_corr, config=G)), md=8),
            ]),
        ]),

        dcc.Tab(label="EDA", value="t-eda", style=TS, selected_style=TA, children=[
            card(
            "NUMERICAL FEATURE — LOG HISTOGRAM & DENSITY BY CHURN",
            html.P("Mirrors 01_eda.py — log1p histogram and smoothed density split by churn label.", style={"color": C_MUTED, "fontSize": "12px", "marginBottom": "12px"}),
            dcc.Dropdown(id="eda-num-col", options=[{"label": c, "value": c} for c in NUM_COLS], value="rev_Mean", style=DD, className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="eda-hist"),    md=6),
                dbc.Col(dcc.Graph(id="eda-density"), md=6),
            ]), accent=C_ACC2),
            dbc.Row([
                dbc.Col(card("BOXPLOT BY CHURN STATUS", dcc.Dropdown(id="eda-box-col", options=[{"label": c, "value": c} for c in NUM_COLS], value="mou_Mean", style=DD, className="mb-3"), dcc.Graph(id="eda-box"), accent=C_ACC2), md=6),
                dbc.Col(card("CATEGORICAL FEATURE BY CHURN", dcc.Dropdown(id="eda-cat-col", options=[{"label": c, "value": c} for c in CAT_COLS], value=CAT_COLS[0], style=DD, className="mb-3"), dcc.Graph(id="eda-cat-bar"), accent=C_ACC2), md=6),
            ]),
        ]),

        dcc.Tab(label="CV Results", value="t-cv", style=TS, selected_style=TA, children=[
            html.P("Loaded from data/cv_results.json (03_baseline_cv.py) and data/tuning_summary.json (04_tuning.py). No models are retrained here.", style={"color": C_MUTED, "fontSize": "12px", "marginTop": "12px", "borderLeft": f"3px solid {C_ACC}", "paddingLeft": "10px"}),
            dbc.Row([
                dbc.Col(card("CV ROC-AUC",  dcc.Graph(figure=fig_cv_roc, config=G)), md=4),
                dbc.Col(card("CV PR-AUC",   dcc.Graph(figure=fig_cv_pr,  config=G)), md=4),
                dbc.Col(card("CV F1",       dcc.Graph(figure=fig_cv_f1,  config=G)), md=4),
            ]),
            card("TUNING SUMMARY — 04_tuning.py", dcc.Graph(figure=fig_tune, config=G)),
            card("CV SCORES TABLE", build_cv_table()),
        ]),

        dcc.Tab(label="Performance", value="t-perf", style=TS, selected_style=TA, children=[
            html.P("Metrics recomputed dynamically from the saved model. Move the threshold slider to see the confusion matrix update live.", style={"color": C_MUTED, "fontSize": "12px", "marginTop": "12px", "borderLeft": f"3px solid {C_ACC2}", "paddingLeft": "10px"}),
            dbc.Row([
                dbc.Col([
                    html.Div("DECISION THRESHOLD", style={"fontFamily": "Space Mono, monospace", "fontSize": "9px", "letterSpacing": "3px", "color": C_MUTED, "marginBottom": "8px"}),
                    dcc.Slider(
                        id="thr-slider", min=0.01, max=0.99, step=0.01, value=OPT_THR,
                        marks={i/10: {"label": f"{i/10:.1f}", "style": {"color": C_MUTED, "fontSize": "10px"}} for i in range(0, 11)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], md=9),
                dbc.Col(html.Div([
                    html.Div("OPTIMAL", style={"fontFamily": "Space Mono, monospace", "fontSize": "9px", "letterSpacing": "2px", "color": C_MUTED}),
                    html.Div(f"{OPT_THR:.3f}", style={"fontFamily": "Space Mono, monospace", "fontSize": "28px", "color": C_ACC}),
                    html.Div("F1-maximising", style={"color": C_MUTED, "fontSize": "11px"}),
                ], style={"background": C_DEEP, "border": f"1px solid {C_BORD}", "borderRadius": "6px", "padding": "12px", "textAlign": "center"}), md=3),
            ], className="mb-3", style={"background": C_CARD, "border": f"1px solid {C_BORD}", "borderRadius": 8, "padding": "20px", "marginBottom": "16px"}),
            html.Div(id="metrics-row"),
            dbc.Row([
                dbc.Col(card("CONFUSION MATRIX", dcc.Graph(id="cm-plot", config=G)), md=4),
                dbc.Col(card("ROC CURVE", dcc.Graph(figure=fig_roc, config=G)), md=4),
                dbc.Col(card("PRECISION-RECALL CURVE", dcc.Graph(figure=fig_pr, config=G)), md=4),
            ]),
        ]),

        dcc.Tab(label="Feature Importance", value="t-fi", style=TS, selected_style=TA, children=[
            html.P(f"Built-in feature importance from the saved {MODEL_NAME} model.", style={"color": C_MUTED, "fontSize": "12px", "marginTop": "12px", "borderLeft": f"3px solid {C_ACC3}", "paddingLeft": "10px"}),
            card(
                f"TOP 20 FEATURE IMPORTANCES  {MODEL_NAME}", 
                dcc.Graph(figure=fig_fi, config=G), 
                accent=C_ACC3
                ),
        ]),

        dcc.Tab(label="Customer Explorer", value="t-cust", style=TS, selected_style=TA, children=[
            dbc.Row([
                dbc.Col(dcc.Dropdown(id="cust-risk",
                    options=[{"label": "All Customers",  "value": "all"},
                             {"label": "🔴 High Risk",   "value": "High Risk"},
                             {"label": "🟡 Medium Risk", "value": "Medium Risk"},
                             {"label": "🟢 Low Risk",    "value": "Low Risk"}],
                    value="all", clearable=False, style=DD), md=3),
                dbc.Col(dcc.Dropdown(id="cust-pred",
                    options=[{"label": "All Predictions", "value": "all"},
                             {"label": "Churn",           "value": "Churn"},
                             {"label": "No Churn",        "value": "No Churn"}],
                    value="all", clearable=False, style=DD), md=3),
                dbc.Col(html.Div(id="cust-count", style={"color": C_MUTED, "fontFamily": "Space Mono, monospace", "fontSize": "11px", "padding": "8px 0"}), md=6),
            ], className="mb-3 mt-2"),
            dcc.Loading(
                dash_table.DataTable(
                    id="cust-table",
                    data=EXPLORER_DF.to_dict("records"),
                    columns=[{"name": c, "id": c, "type": "numeric" if c == "Churn Prob" else "text"} for c in EXPLORER_DF.columns],
                    page_size=20, page_action="native",
                    sort_action="native", filter_action="native",
                    row_selectable="single", selected_rows=[],
                    export_format="csv",
                    style_table={"overflowX": "auto", "borderRadius": "6px"},
                    style_cell={"backgroundColor": C_CARD, "color": "#B0BEC5", "border": f"1px solid {C_BORD}", "padding": "10px 14px", "fontFamily": "DM Sans, sans-serif", "fontSize": "13px", "textAlign": "left", "minWidth": "80px"},
                    style_header={"backgroundColor": C_DEEP, "color": C_ACC, "fontFamily": "Space Mono, monospace", "fontSize": "10px", "letterSpacing": "1px", "border": f"1px solid {C_BORD}", "fontWeight": "700"},
                    style_data_conditional=TABLE_COND,
                ), color=C_ACC, type="dot",
            ),
        ]),
    ]),
], fluid=True, style={"backgroundColor": C_BG, "minHeight": "100vh", "padding": "0 28px 40px"})

# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output("eda-hist",    "figure"),
    Output("eda-density", "figure"),
    Input("eda-num-col",  "value"),
)
def eda_num(col):
    fig_h = go.Figure()
    fig_d = go.Figure()
    for cv, label, color in [(0, "No Churn", C_GREEN), (1, "Churn", C_RED)]:
        vals = pd.to_numeric(RAW_DF.loc[RAW_DF["churn"] == cv, col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        fig_h.add_trace(go.Histogram(
            x=np.log1p(vals), name=label, opacity=0.7,
            marker_color=color, nbinsx=50))
        clipped = vals.clip(upper=float(np.percentile(vals, 97)))
        counts, edges = np.histogram(clipped, bins=80, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        smooth  = np.convolve(counts, np.ones(7)/7, mode="same")
        fill_c  = "rgba(0,200,83,0.12)" if color == C_GREEN else "rgba(255,61,87,0.12)"
        fig_d.add_trace(go.Scatter(x=centers, y=smooth, mode="lines", name=label,
            line=dict(color=color, width=2), fill="tozeroy", fillcolor=fill_c))
    fig_h.update_layout(**BL, barmode="overlay",
        title=f"log1p({col}) — Histogram by Churn", height=340)
    fig_d.update_layout(**BL,
        title=f"{col} — Density by Churn", height=340)
    return fig_h, fig_d

@app.callback(Output("eda-box", "figure"), Input("eda-box-col", "value"))
def eda_box(col):
    fig = go.Figure()
    for cv, label, color in [(0, "No Churn", C_GREEN), (1, "Churn", C_RED)]:
        vals = pd.to_numeric(RAW_DF.loc[RAW_DF["churn"] == cv, col], errors="coerce").dropna()
        fig.add_trace(go.Box(y=vals, name=label, marker_color=color, boxmean=True))
    fig.update_layout(**BL, title=f"{col} — Boxplot by Churn", height=380)
    return fig

# [FIX] Added the missing Categorical Plot Callback
@app.callback(Output("eda-cat-bar", "figure"), Input("eda-cat-col", "value"))
def eda_cat(col):
    if col not in RAW_DF.columns:
        return go.Figure().update_layout(**BL, title="Column not found")
        
    df_counts = RAW_DF.groupby([col, 'churn']).size().reset_index(name='count')
    df_counts['churn_label'] = df_counts['churn'].map({0: 'No Churn', 1: 'Churn'})
    
    fig = px.bar(
        df_counts, x=col, y='count', color='churn_label',
        barmode='group', color_discrete_map={'No Churn': C_GREEN, 'Churn': C_RED}
    )
    fig.update_layout(**BL, title=f"{col} — Distribution by Churn", height=380)
    return fig

# [FIX] Added missing callback for the Performance Tab (Threshold Slider, Metrics, CM)
@app.callback(
    Output("metrics-row", "children"),
    Output("cm-plot", "figure"),
    Input("thr-slider", "value")
)
def update_perf(thr):
    yp = (PROB_TEST >= thr).astype(int)
    
    acc = accuracy_score(Y_TEST, yp)
    prec = precision_score(Y_TEST, yp, zero_division=0)
    rec = recall_score(Y_TEST, yp, zero_division=0)
    f1 = f1_score(Y_TEST, yp, zero_division=0)
    
    metrics_ui = dbc.Row([
        kpi("ACCURACY", f"{acc:.4f}", color=C_ACC),
        kpi("PRECISION", f"{prec:.4f}", color=C_ACC2),
        kpi("RECALL", f"{rec:.4f}", color=C_ACC3),
        kpi("F1 SCORE", f"{f1:.4f}", color=C_AMBER),
    ])
    
    cm = confusion_matrix(Y_TEST, yp)
    fig_cm = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True Label"),
        x=["No Churn", "Churn"], y=["No Churn", "Churn"]
    )
    fig_cm.update_layout(**BL, title=f"Confusion Matrix (Threshold = {thr:.2f})", height=400)
    
    return metrics_ui, fig_cm

# [FIX] Added missing callback for the Customer Explorer Data Table
@app.callback(
    Output("cust-table", "data"),
    Output("cust-count", "children"),
    Input("cust-risk", "value"),
    Input("cust-pred", "value")
)
def update_table(risk, pred):
    dff = EXPLORER_DF.copy()
    if risk != "all":
        dff = dff[dff["Risk Level"] == risk]
    if pred != "all":
        dff = dff[dff["Predicted"] == pred]
    
    count_text = f"Showing {len(dff):,} of {len(EXPLORER_DF):,} customers."
    return dff.to_dict("records"), count_text

# [FIX] Added missing execution block to run the app
if __name__ == "__main__":
    app.run(debug=True, port=8050)