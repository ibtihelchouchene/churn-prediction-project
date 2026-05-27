"""pages/customers.py — Customer Explorer Page."""

import dash
from dash import html, dcc

dash.register_page(__name__, path="/customers", name="Customer Explorer")

layout = html.Div([
    html.Div([
        html.Div("CUSTOMER EXPLORER", style={
            "fontFamily": "Space Mono, monospace", "fontSize": "10px",
            "letterSpacing": "4px", "color": "#546E7A",
        }),
        html.H2("Interactive Customer Analysis", style={
            "fontFamily": "Space Mono, monospace", "fontSize": "28px",
            "color": "#E0E6ED", "margin": "6px 0 4px",
        }),
        html.Div("Browse and analyze individual customer churn risk profiles", style={"color": "#546E7A", "fontSize": "13px"}),
    ], style={"marginBottom": "32px"}),

    html.Div([
        html.Div("Customer exploration table coming soon...", style={"color": "#B0BEC5", "fontSize": "14px", "padding": "40px", "textAlign": "center"})
    ], style={"backgroundColor": "#0F2035", "borderRadius": "8px", "padding": "32px"}),

], style={"padding": "32px"})
