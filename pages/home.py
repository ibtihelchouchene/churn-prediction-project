"""pages/home.py — Executive Overview Dashboard Page."""

import dash
from dash import html, dcc

dash.register_page(__name__, path="/", name="Executive Overview")

layout = html.Div([
    html.Div([
        html.Div("EXECUTIVE OVERVIEW", style={
            "fontFamily": "Space Mono, monospace", "fontSize": "10px",
            "letterSpacing": "4px", "color": "#546E7A",
        }),
        html.H2("Churn Analytics Dashboard", style={
            "fontFamily": "Space Mono, monospace", "fontSize": "28px",
            "color": "#E0E6ED", "margin": "6px 0 4px",
        }),
        html.Div("Real-time customer churn predictions and risk analysis", style={"color": "#546E7A", "fontSize": "13px"}),
    ], style={"marginBottom": "32px"}),

    html.Div([
        html.Div([
            html.Div("Total Customers", style={"fontSize": "12px", "color": "#546E7A"}),
            html.Div("100,000", style={"fontSize": "32px", "fontWeight": "700", "color": "#00E5FF"}),
        ], style={"padding": "24px", "backgroundColor": "#0F2035", "borderRadius": "8px", "borderLeft": "3px solid #00E5FF"}),
        
        html.Div([
            html.Div("Churn Rate", style={"fontSize": "12px", "color": "#546E7A"}),
            html.Div("49.56%", style={"fontSize": "32px", "fontWeight": "700", "color": "#FF3D57"}),
        ], style={"padding": "24px", "backgroundColor": "#0F2035", "borderRadius": "8px", "borderLeft": "3px solid #FF3D57"}),
        
        html.Div([
            html.Div("Model Accuracy", style={"fontSize": "12px", "color": "#546E7A"}),
            html.Div("67.45%", style={"fontSize": "32px", "fontWeight": "700", "color": "#00E5FF"}),
        ], style={"padding": "24px", "backgroundColor": "#0F2035", "borderRadius": "8px", "borderLeft": "3px solid #00E5FF"}),
    ], style={
        "display": "grid",
        "gridTemplateColumns": "repeat(3, 1fr)",
        "gap": "16px",
        "marginBottom": "32px",
    }),

    html.Div([
        html.Div("Dashboard is loading...", style={"color": "#B0BEC5", "fontSize": "14px", "padding": "40px", "textAlign": "center"})
    ], style={"backgroundColor": "#0F2035", "borderRadius": "8px", "padding": "32px"}),

], style={"padding": "32px"})
