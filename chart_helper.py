"""
Chart selection logic for the Spend Insights Assistant.

Given the user's question and the underlying transaction dataframe, decides
whether a chart would help illustrate the answer, and if so, which kind
(monthly trend / category breakdown / customer segmentation) and builds it
with Plotly. Charts are built directly from the raw data (not parsed from
the LLM's text answer), so the numbers shown are always exactly correct.
"""

import re
import pandas as pd
import plotly.graph_objects as go

# Colors pulled from the same palette used in the dashboard, for consistency
COLOR_PRIMARY = "#4f7cff"
COLOR_SECONDARY = "#7c5cff"
COLOR_SEGMENT = ["#1D9E75", "#378ADD", "#888780"]  # High / Medium / Low

CATEGORY_KEYWORDS = ["category", "categories", "merchant", "spend by", "breakdown",
                     "which category", "top category", "highest spend"]
TREND_KEYWORDS = ["trend", "over time", "monthly", "month", "quarter", "last quarter",
                   "growth", "increase", "decrease", "pattern"]
SEGMENT_KEYWORDS = ["segment", "segmentation", "high value", "low value", "medium value",
                     "customer group", "customer type"]


def _matches_any(text: str, keywords: list) -> bool:
    text = text.lower()
    return any(kw in text for kw in keywords)


def pick_chart_type(question: str) -> str | None:
    """Return 'trend', 'category', 'segment', or None if no chart fits."""
    if _matches_any(question, TREND_KEYWORDS):
        return "trend"
    if _matches_any(question, CATEGORY_KEYWORDS):
        return "category"
    if _matches_any(question, SEGMENT_KEYWORDS):
        return "segment"
    return None


def build_trend_chart(df: pd.DataFrame) -> go.Figure:
    # Derive month_sort on the fly if it's not already in the dataframe —
    # this lets the chart work on the plain transactions.csv, not just the
    # enriched transactions_dashboard.csv.
    if "month_sort" not in df.columns:
        df = df.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        df["month_sort"] = df["transaction_date"].dt.strftime("%Y-%m")

    monthly = df.groupby("month_sort")["amount"].sum().sort_index().reset_index()
    fig = go.Figure(go.Scatter(
        x=monthly["month_sort"], y=monthly["amount"],
        mode="lines+markers", line=dict(color=COLOR_PRIMARY, width=3),
        marker=dict(size=7, color=COLOR_PRIMARY),
        fill="tozeroy", fillcolor="rgba(79,124,255,0.12)",
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="Monthly spend trend",
        margin=dict(l=10, r=10, t=40, b=10),
        height=280,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(tickprefix="$", gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    )
    return fig


def build_category_chart(df: pd.DataFrame) -> go.Figure:
    cat = df.groupby("merchant_category")["amount"].sum().sort_values(ascending=True).reset_index()
    fig = go.Figure(go.Bar(
        x=cat["amount"], y=cat["merchant_category"], orientation="h",
        marker=dict(color=COLOR_SECONDARY),
        hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="Spend by merchant category",
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickprefix="$", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    )
    return fig


def build_segment_chart(df: pd.DataFrame) -> go.Figure:
    if "customer_segment" not in df.columns:
        return None
    seg = df.groupby("customer_segment")["customer_id"].nunique().reset_index()
    seg.columns = ["segment", "customers"]
    fig = go.Figure(go.Pie(
        labels=seg["segment"], values=seg["customers"], hole=0.6,
        marker=dict(colors=COLOR_SEGMENT),
        hovertemplate="%{label}<br>%{value} customers<extra></extra>",
    ))
    fig.update_layout(
        title="Customers by segment",
        margin=dict(l=10, r=10, t=40, b=10),
        height=280,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
    )
    return fig


def maybe_build_chart(question: str, df: pd.DataFrame):
    """Main entry point: returns a Plotly figure if the question warrants
    one, or None if a chart wouldn't add value (or can't be built safely)."""
    chart_type = pick_chart_type(question)
    try:
        if chart_type == "trend":
            return build_trend_chart(df)
        if chart_type == "category":
            return build_category_chart(df)
        if chart_type == "segment":
            return build_segment_chart(df)
    except Exception:
        # If anything about the data prevents building this chart, fail
        # quietly — the text answer still comes through either way.
        return None
    return None
