"""
Plotly Chart Generation Engine for DataInsight AI.
Provides executive-grade, responsive charts with clean SaaS typography,
positive/negative color semantics, custom hover tooltips, and plain-language explanations.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.analytics import ColumnMapping, BusinessKPIs
from utils.helper import format_currency, format_percent


# Modern SaaS Color Palette
COLORS = {
    "primary": "#2563EB",       # Professional Blue
    "secondary": "#6366F1",     # Indigo
    "positive": "#10B981",      # Emerald Green
    "negative": "#EF4444",      # Rose Red
    "warning": "#F59E0B",       # Amber
    "neutral": "#64748B",       # Slate
    "light_bg": "#F8FAFC",      # Clean canvas
    "grid": "#E2E8F0",          # Light border
    "text_dark": "#0F172A",     # Deep Slate
    "palette": ["#2563EB", "#10B981", "#F59E0B", "#8B5CF6", "#06B6D4", "#EC4899", "#F97316"]
}


def _apply_saas_theme(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """Applies a clean, commercial SaaS theme to any Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "x": 0.01,
            "y": 0.95,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif", "color": COLORS["text_dark"]}
        },
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=45, r=30, t=55, b=45),
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=13,
            font_family="Inter, sans-serif",
            font_color="#FFFFFF"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color=COLORS["neutral"])
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=COLORS["grid"],
            gridwidth=1,
            linecolor=COLORS["grid"],
            tickfont=dict(size=11, color=COLORS["neutral"]),
            title_font=dict(size=12, color=COLORS["text_dark"])
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS["grid"],
            gridwidth=1,
            linecolor=COLORS["grid"],
            tickfont=dict(size=11, color=COLORS["neutral"]),
            title_font=dict(size=12, color=COLORS["text_dark"])
        )
    )
    return fig


# ==============================================================================
# 1. Executive Performance Trend Chart (Date + Financial)
# ==============================================================================

def create_kpi_trend_chart(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    currency_symbol: str = "$"
) -> Optional[Tuple[go.Figure, str]]:
    """
    Creates a dual-metric time-series trend of Revenue and Profit.
    Returns (figure, plain_language_explanation).
    """
    if df is None or df.empty or not mapping.date_col or mapping.date_col not in df.columns:
        return None

    date_series = pd.to_datetime(df[mapping.date_col], errors="coerce")
    valid_mask = date_series.notna()
    if valid_mask.sum() < 2:
        return None

    plot_df = df[valid_mask].copy()
    plot_df["_date"] = date_series[valid_mask]
    
    rev_col = mapping.revenue_col
    cost_col = mapping.cost_col
    profit_col = mapping.profit_col

    # Calculate profit series
    if mapping.profit_mode == "direct" and profit_col and profit_col in plot_df.columns:
        plot_df["_profit"] = pd.to_numeric(plot_df[profit_col], errors="coerce").fillna(0)
    elif rev_col and cost_col and rev_col in plot_df.columns and cost_col in plot_df.columns:
        r = pd.to_numeric(plot_df[rev_col], errors="coerce").fillna(0)
        c = pd.to_numeric(plot_df[cost_col], errors="coerce").fillna(0)
        plot_df["_profit"] = r - c
    elif profit_col and profit_col in plot_df.columns:
        plot_df["_profit"] = pd.to_numeric(plot_df[profit_col], errors="coerce").fillna(0)
    else:
        plot_df["_profit"] = None

    if rev_col and rev_col in plot_df.columns:
        plot_df["_revenue"] = pd.to_numeric(plot_df[rev_col], errors="coerce").fillna(0)
    else:
        plot_df["_revenue"] = None

    if plot_df["_revenue"] is None and plot_df["_profit"] is None:
        return None

    # Resampling granularity
    min_date = plot_df["_date"].min()
    max_date = plot_df["_date"].max()
    span_days = (max_date - min_date).days

    if span_days > 180:
        freq = "ME"
        date_format = "%b %Y"
        freq_label = "Monthly"
    elif span_days > 30:
        freq = "W"
        date_format = "%d %b %Y"
        freq_label = "Weekly"
    else:
        freq = "D"
        date_format = "%d %b"
        freq_label = "Daily"

    plot_df = plot_df.set_index("_date")
    agg_rules = {}
    if plot_df["_revenue"] is not None:
        agg_rules["_revenue"] = "sum"
    if plot_df["_profit"] is not None:
        agg_rules["_profit"] = "sum"

    resampled = plot_df.resample(freq).agg(agg_rules).reset_index()
    resampled["_formatted_date"] = resampled["_date"].dt.strftime(date_format)

    fig = go.Figure()

    if "_revenue" in resampled.columns:
        fig.add_trace(go.Scatter(
            x=resampled["_formatted_date"],
            y=resampled["_revenue"],
            name="Revenue",
            mode="lines+markers",
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(size=6, color=COLORS["primary"]),
            hovertemplate="<b>%{x}</b><br>Revenue: " + currency_symbol + "%{y:,.2f}<extra></extra>"
        ))

    if "_profit" in resampled.columns and resampled["_profit"].notna().any():
        fig.add_trace(go.Scatter(
            x=resampled["_formatted_date"],
            y=resampled["_profit"],
            name="Net Profit",
            mode="lines+markers",
            line=dict(color=COLORS["positive"], width=2.5, dash="dash"),
            marker=dict(size=6, color=COLORS["positive"]),
            hovertemplate="<b>%{x}</b><br>Net Profit: " + currency_symbol + "%{y:,.2f}<extra></extra>"
        ))

    fig = _apply_saas_theme(fig, title=f"📈 Business Performance Trend ({freq_label})")
    fig.update_xaxes(title_text="Date Period")
    fig.update_yaxes(title_text=f"Amount ({currency_symbol})")

    explanation = (
        f"This chart displays the {freq_label.lower()} trajectory of company revenue and net profit over time. "
        "Divergence between revenue (blue) and net profit (green) indicates fluctuating cost structures or changing product mix."
    )
    return fig, explanation


# ==============================================================================
# 2. Top Products Profitability Ranking (Diverging Positive / Negative Bar)
# ==============================================================================

def create_product_profitability_chart(
    product_df: pd.DataFrame,
    currency_symbol: str = "$",
    max_items: int = 12
) -> Optional[Tuple[go.Figure, str]]:
    """
    Creates a diverging bar chart ranking top and bottom products by Net Profit.
    Positive profits are colored green; negative profits (losses) are colored red.
    """
    if product_df is None or product_df.empty or "Net Profit" not in product_df.columns:
        return None

    valid_df = product_df[product_df["Net Profit"].notna()].copy()
    if valid_df.empty:
        return None

    # Pick top profitable and bottom loss-making items
    top_items = valid_df[valid_df["Net Profit"] > 0].head(max_items // 2)
    bottom_items = valid_df[valid_df["Net Profit"] < 0].tail(max_items // 2)
    
    if bottom_items.empty:
        # If no loss-makers, just show top products
        plot_items = valid_df.head(max_items)
    else:
        plot_items = pd.concat([top_items, bottom_items]).drop_duplicates(subset=["Product"])
        plot_items = plot_items.sort_values(by="Net Profit", ascending=True)

    colors = [COLORS["positive"] if p >= 0 else COLORS["negative"] for p in plot_items["Net Profit"]]

    fig = go.Figure(go.Bar(
        x=plot_items["Net Profit"],
        y=plot_items["Product"],
        orientation="h",
        marker_color=colors,
        text=[format_currency(val, currency_symbol, decimals=0, abbreviate=True) for val in plot_items["Net Profit"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Net Profit: " + currency_symbol + "%{x:,.2f}<br>" +
            "<extra></extra>"
        )
    ))

    fig = _apply_saas_theme(fig, title="🏆 Product Profitability Ranking (Gainers & Loss-Makers)")
    fig.update_xaxes(title_text=f"Net Profit / Loss ({currency_symbol})", zeroline=True, zerolinecolor="#94A3B8", zerolinewidth=1.5)
    fig.update_yaxes(title_text="Product", automargin=True)

    loss_count = (plot_items["Net Profit"] < 0).sum()
    if loss_count > 0:
        explanation = (
            f"Highlights leading profit contributors (green) alongside loss-making products (red). "
            f"{loss_count} product(s) shown generate negative net profit and represent immediate candidates for pricing or cost restructuring."
        )
    else:
        explanation = (
            "Displays the top-ranking products by net cumulative profit. All leading products demonstrate positive net contribution."
        )

    return fig, explanation


# ==============================================================================
# 3. Revenue vs Profit Scatter Matrix (Margin Quadrants)
# ==============================================================================

def create_revenue_vs_profit_chart(
    product_df: pd.DataFrame,
    currency_symbol: str = "$"
) -> Optional[Tuple[go.Figure, str]]:
    """
    Creates a Revenue vs Net Profit scatter visualization.
    Allows business users to immediately spot high-revenue low-margin items.
    """
    if product_df is None or product_df.empty or "Net Profit" not in product_df.columns or "Revenue" not in product_df.columns:
        return None

    plot_df = product_df[product_df["Revenue"].notna() & product_df["Net Profit"].notna()].copy()
    if plot_df.empty:
        return None

    # Size points by Transactions or Units Sold
    size_col = "Transactions" if "Transactions" in plot_df.columns else None

    # Status color map
    status_colors = {
        "Loss-Maker 🔴": COLORS["negative"],
        "High Rev / Low Margin ⚠️": COLORS["warning"],
        "Top Performer 🟢": COLORS["positive"],
        "Profitable 🔵": COLORS["primary"],
        "Active Volume ⚪": COLORS["neutral"],
        "Low Volume ⚪": "#94A3B8"
    }
    plot_df["color"] = plot_df["Status"].map(lambda s: status_colors.get(s, COLORS["primary"]))

    fig = go.Figure()

    for status_label, group in plot_df.groupby("Status"):
        fig.add_trace(go.Scatter(
            x=group["Revenue"],
            y=group["Net Profit"],
            mode="markers",
            name=status_label,
            marker=dict(
                color=status_colors.get(status_label, COLORS["primary"]),
                size=np.clip(group["Transactions"] * 1.5, 9, 28) if size_col else 11,
                opacity=0.85,
                line=dict(width=1, color="#FFFFFF")
            ),
            text=group["Product"],
            customdata=np.stack((group["Product"], group["Profit Margin (%)"].fillna(0), group["Transactions"]), axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>" +
                "Revenue: " + currency_symbol + "%{x:,.2f}<br>" +
                "Net Profit: " + currency_symbol + "%{y:,.2f}<br>" +
                "Profit Margin: %{customdata[1]:.1f}%<br>" +
                "Transactions: %{customdata[2]:,.0f}<br>" +
                "<extra></extra>"
            )
        ))

    # Breakeven line at y = 0
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["negative"], opacity=0.7, annotation_text="Breakeven", annotation_position="bottom right")

    fig = _apply_saas_theme(fig, title="⚖️ Revenue vs. Profit Analysis (Product Economics)")
    fig.update_xaxes(title_text=f"Total Revenue ({currency_symbol})")
    fig.update_yaxes(title_text=f"Net Profit ({currency_symbol})")

    explanation = (
        "Each bubble represents a product. Items plotted below the horizontal dashed line generate net losses. "
        "Items located far to the right but close to zero profit indicate high-revenue products with weak operating margins."
    )
    return fig, explanation


# ==============================================================================
# 4. Category Performance Breakdown (Bar or Donut)
# ==============================================================================

def create_category_performance_chart(
    category_df: pd.DataFrame,
    currency_symbol: str = "$"
) -> Optional[Tuple[go.Figure, str]]:
    """Creates a category performance breakdown chart."""
    if category_df is None or category_df.empty or "Category" not in category_df.columns:
        return None

    plot_df = category_df.head(10).copy()

    fig = go.Figure()

    # Revenue bar
    if "Revenue" in plot_df.columns:
        fig.add_trace(go.Bar(
            x=plot_df["Category"],
            y=plot_df["Revenue"],
            name="Revenue",
            marker_color=COLORS["primary"],
            hovertemplate="<b>%{x}</b><br>Revenue: " + currency_symbol + "%{y:,.2f}<extra></extra>"
        ))

    # Profit bar (if available)
    if "Net Profit" in plot_df.columns and plot_df["Net Profit"].notna().any():
        fig.add_trace(go.Bar(
            x=plot_df["Category"],
            y=plot_df["Net Profit"],
            name="Net Profit",
            marker_color=[COLORS["positive"] if p >= 0 else COLORS["negative"] for p in plot_df["Net Profit"]],
            hovertemplate="<b>%{x}</b><br>Net Profit: " + currency_symbol + "%{y:,.2f}<extra></extra>"
        ))

    fig = _apply_saas_theme(fig, title="📁 Category Performance Comparison")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Category", tickangle=-25)
    fig.update_yaxes(title_text=f"Amount ({currency_symbol})")

    explanation = (
        "Compares commercial volume (Revenue) and bottom-line return (Net Profit) across major product categories. "
        "Helps identify which departments drive the highest profit efficiency."
    )
    return fig, explanation


# ==============================================================================
# 5. Non-Financial Fallback Charts (Distributions & Categorical Counts)
# ==============================================================================

def create_categorical_distribution_chart(
    df: pd.DataFrame,
    cat_col: str,
    top_n: int = 10
) -> Optional[Tuple[go.Figure, str]]:
    """Creates a frequency bar chart for general/non-financial categorical columns."""
    if df is None or df.empty or cat_col not in df.columns:
        return None

    counts = df[cat_col].value_counts().head(top_n).reset_index()
    counts.columns = [cat_col, "Count"]

    fig = go.Figure(go.Bar(
        x=counts[cat_col].astype(str),
        y=counts["Count"],
        marker_color=COLORS["primary"],
        text=counts["Count"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Records: %{y:,}<extra></extra>"
    ))

    fig = _apply_saas_theme(fig, title=f"📊 Distribution of Records by {cat_col}")
    fig.update_xaxes(title_text=cat_col, tickangle=-20)
    fig.update_yaxes(title_text="Record Count")

    explanation = f"Shows the record distribution across the top {len(counts)} categories for column '{cat_col}'."
    return fig, explanation


def create_numeric_distribution_chart(
    df: pd.DataFrame,
    num_col: str
) -> Optional[Tuple[go.Figure, str]]:
    """Creates a histogram / distribution curve for numeric columns in general datasets."""
    if df is None or df.empty or num_col not in df.columns:
        return None

    clean_series = pd.to_numeric(df[num_col], errors="coerce").dropna()
    if clean_series.empty:
        return None

    fig = px.histogram(
        clean_series,
        x=num_col,
        nbins=25,
        color_discrete_sequence=[COLORS["primary"]],
        marginal="box"
    )

    fig = _apply_saas_theme(fig, title=f"📊 Distribution & Outlier Analysis: {num_col}")
    fig.update_xaxes(title_text=num_col)
    fig.update_yaxes(title_text="Frequency")

    explanation = (
        f"Histogram and box plot illustrating the distribution spread, central tendency, "
        f"and potential outliers for variable '{num_col}'."
    )
    return fig, explanation


# ==============================================================================
# 6. Advanced Custom Visualization Studio Builder
# ==============================================================================

def create_custom_studio_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: Optional[str] = None,
    color_col: Optional[str] = None,
    agg_func: str = "sum",
    title: str = "",
    height: int = 500
) -> go.Figure:
    """
    Renders customizable advanced charts for analysts with built-in validation
    and defensive aggregation handling.
    """
    clean_df = df.copy()

    # Pre-aggregation for bar and line when requested
    if y_col and y_col in clean_df.columns and pd.api.types.is_numeric_dtype(clean_df[y_col]):
        clean_df[y_col] = pd.to_numeric(clean_df[y_col], errors="coerce")

    # Map Aggregation Function
    agg_map = {
        "sum": "sum",
        "mean": "mean",
        "count": "count",
        "median": "median",
        "min": "min",
        "max": "max"
    }
    selected_agg = agg_map.get(agg_func.lower(), "sum")

    fig: go.Figure

    if chart_type == "Bar Chart":
        if y_col and y_col != "None":
            # Grouped aggregation
            group_cols = [x_col]
            if color_col and color_col != "None" and color_col in clean_df.columns:
                group_cols.append(color_col)
            
            agg_df = clean_df.groupby(group_cols, as_index=False)[y_col].agg(selected_agg)
            fig = px.bar(
                agg_df,
                x=x_col,
                y=y_col,
                color=color_col if color_col != "None" else None,
                color_discrete_sequence=COLORS["palette"]
            )
        else:
            fig = px.bar(
                clean_df[x_col].value_counts().reset_index(),
                x=x_col,
                y="count",
                color_discrete_sequence=[COLORS["primary"]]
            )

    elif chart_type == "Horizontal Bar":
        if y_col and y_col != "None":
            group_cols = [x_col]
            if color_col and color_col != "None" and color_col in clean_df.columns:
                group_cols.append(color_col)
            agg_df = clean_df.groupby(group_cols, as_index=False)[y_col].agg(selected_agg)
            fig = px.bar(
                agg_df,
                y=x_col,
                x=y_col,
                orientation="h",
                color=color_col if color_col != "None" else None,
                color_discrete_sequence=COLORS["palette"]
            )
        else:
            fig = px.bar(
                clean_df[x_col].value_counts().reset_index(),
                y=x_col,
                x="count",
                orientation="h",
                color_discrete_sequence=[COLORS["primary"]]
            )

    elif chart_type == "Line Chart":
        if y_col and y_col != "None":
            group_cols = [x_col]
            if color_col and color_col != "None" and color_col in clean_df.columns:
                group_cols.append(color_col)
            agg_df = clean_df.groupby(group_cols, as_index=False)[y_col].agg(selected_agg)
            fig = px.line(
                agg_df,
                x=x_col,
                y=y_col,
                color=color_col if color_col != "None" else None,
                markers=True,
                color_discrete_sequence=COLORS["palette"]
            )
        else:
            fig = px.line(clean_df, x=x_col, color_discrete_sequence=[COLORS["primary"]])

    elif chart_type == "Area Chart":
        if y_col and y_col != "None":
            group_cols = [x_col]
            if color_col and color_col != "None" and color_col in clean_df.columns:
                group_cols.append(color_col)
            agg_df = clean_df.groupby(group_cols, as_index=False)[y_col].agg(selected_agg)
            fig = px.area(
                agg_df,
                x=x_col,
                y=y_col,
                color=color_col if color_col != "None" else None,
                color_discrete_sequence=COLORS["palette"]
            )
        else:
            fig = px.area(clean_df, x=x_col, color_discrete_sequence=[COLORS["primary"]])

    elif chart_type == "Scatter Plot":
        fig = px.scatter(
            clean_df,
            x=x_col,
            y=y_col,
            color=color_col if color_col != "None" else None,
            color_discrete_sequence=COLORS["palette"],
            opacity=0.8
        )

    elif chart_type == "Box Plot":
        fig = px.box(
            clean_df,
            x=x_col,
            y=y_col if y_col != "None" else None,
            color=color_col if color_col != "None" else None,
            color_discrete_sequence=COLORS["palette"]
        )

    elif chart_type == "Histogram":
        fig = px.histogram(
            clean_df,
            x=x_col,
            y=y_col if y_col != "None" else None,
            color=color_col if color_col != "None" else None,
            color_discrete_sequence=COLORS["palette"]
        )

    elif chart_type == "Donut / Pie Chart":
        val_col = y_col if (y_col and y_col != "None") else None
        fig = px.pie(
            clean_df,
            names=x_col,
            values=val_col,
            hole=0.45,
            color_discrete_sequence=COLORS["palette"]
        )

    elif chart_type == "Correlation Heatmap":
        num_df = clean_df.select_dtypes(include="number")
        if num_df.shape[1] >= 2:
            corr = num_df.corr().round(2)
            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="Blues",
                aspect="auto"
            )
        else:
            fig = go.Figure()
            fig.add_annotation(text="At least 2 numeric columns required for Heatmap.", showarrow=False)

    else:
        fig = px.bar(clean_df, x=x_col, y=y_col, color_discrete_sequence=[COLORS["primary"]])

    chart_title = title if title.strip() else f"{chart_type}: {x_col}" + (f" vs {y_col}" if y_col and y_col != "None" else "")
    return _apply_saas_theme(fig, title=chart_title, height=height)
