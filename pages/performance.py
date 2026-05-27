"""pages/performance.py — Model Performance Page."""

import dash
from dash import html, dcc

dash.register_page(__name__, path="/performance", name="Model Performance")

layout = html.Div([
    html.Div([
        html.Div("MODEL PERFORMANCE", style={
            "fontFamily": "Space Mono, monospace", "fontSize": "10px",
            "letterSpacing": "4px", "color": "#546E7A",
        }),
        html.H2("Evaluation Report", style={
            "fontFamily": "Space Mono, monospace", "fontSize": "28px",
            "color": "#E0E6ED", "margin": "6px 0 4px",
        }),
        html.Div("Model metrics, ROC curves, and feature importance", style={"color": "#546E7A", "fontSize": "13px"}),
    ], style={"marginBottom": "32px"}),

    html.Div([
        html.Div([
            html.Div("ROC-AUC Score", style={"fontSize": "12px", "color": "#546E7A"}),
            html.Div("0.6745", style={"fontSize": "32px", "fontWeight": "700", "color": "#4DD0E1"}),
        ], style={"padding": "24px", "backgroundColor": "#0F2035", "borderRadius": "8px", "borderLeft": "3px solid #4DD0E1"}),
        
        html.Div([
            html.Div("PR-AUC Score", style={"fontSize": "12px", "color": "#546E7A"}),
            html.Div("0.6591", style={"fontSize": "32px", "fontWeight": "700", "color": "#4DD0E1"}),
        ], style={"padding": "24px", "backgroundColor": "#0F2035", "borderRadius": "8px", "borderLeft": "3px solid #4DD0E1"}),
        
        html.Div([
            html.Div("F1 Score", style={"fontSize": "12px", "color": "#546E7A"}),
            html.Div("0.6796", style={"fontSize": "32px", "fontWeight": "700", "color": "#4DD0E1"}),
        ], style={"padding": "24px", "backgroundColor": "#0F2035", "borderRadius": "8px", "borderLeft": "3px solid #4DD0E1"}),
    ], style={
        "display": "grid",
        "gridTemplateColumns": "repeat(3, 1fr)",
        "gap": "16px",
        "marginBottom": "32px",
    }),

    html.Div([
        html.Div("Performance visualizations coming soon...", style={"color": "#B0BEC5", "fontSize": "14px", "padding": "40px", "textAlign": "center"})
    ], style={"backgroundColor": "#0F2035", "borderRadius": "8px", "padding": "32px"}),

], style={"padding": "32px"})
