"""
DataInsight AI — Business Performance Dashboard
Executive SaaS Business Intelligence Dashboard with automatic dataset understanding,
financial KPI calculations, executive visual analytics, product profitability,
actionable recommendations, and an advanced analyst studio.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO

from utils.analytics import (
    ColumnMapping,
    detect_column_mapping,
    calculate_business_kpis,
    analyze_product_performance,
    analyze_category_performance,
    generate_business_recommendations,
    BusinessAlert,
    AlertSystemResult,
    detect_loss_making_products
)
from utils.charts import (
    create_kpi_trend_chart,
    create_product_profitability_chart,
    create_revenue_vs_profit_chart,
    create_category_performance_chart,
    create_categorical_distribution_chart,
    create_numeric_distribution_chart,
    create_custom_studio_chart
)
from utils.ai_insights import generate_executive_ai_briefing, generate_alert_investigation_briefing
from utils.helper import format_currency, format_percent, format_number


# Page Configuration
st.set_page_config(
    page_title="Business Dashboard | DataInsight AI",
    page_icon="💼",
    layout="wide"
)

# Custom Styling for Commercial SaaS look
st.markdown("""
<style>
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #0F172A;
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 4px;
    }
    .rec-card {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 14px;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }
    .rec-high {
        border-left-color: #EF4444 !important;
    }
    .rec-opp {
        border-left-color: #10B981 !important;
    }
    .rec-med {
        border-left-color: #F59E0B !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Helper: Sample Dataset Generator for Instant Demo
# ==============================================================================
def load_sample_ecommerce_data() -> pd.DataFrame:
    """Generates a realistic commercial e-commerce dataset for instant testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=120, freq="D")
    
    products = [
        {"name": "Ultra Wireless Headphones", "category": "Audio", "price": 149.99, "cost": 75.00},
        {"name": "Ergonomic Mechanical Keyboard", "category": "Peripherals", "price": 129.50, "cost": 60.00},
        {"name": "Noise-Cancelling Earbuds", "category": "Audio", "price": 89.00, "cost": 42.00},
        {"name": "Pro 4K Gaming Monitor", "category": "Displays", "price": 389.00, "cost": 310.00},
        {"name": "Smart Fitness Watch", "category": "Wearables", "price": 199.00, "cost": 110.00},
        {"name": "USB-C Fast Charging Hub", "category": "Accessories", "price": 35.00, "cost": 33.50},  # Low margin
        {"name": "Budget Bluetooth Speaker", "category": "Audio", "price": 29.99, "cost": 34.00},      # Loss-maker!
        {"name": "Precision Optical Mouse", "category": "Peripherals", "price": 45.00, "cost": 18.00},
        {"name": "Alloy Laptop Stand", "category": "Accessories", "price": 39.99, "cost": 16.00},
        {"name": "Clearance Screen Protector", "category": "Accessories", "price": 9.99, "cost": 12.50} # Loss-maker!
    ]
    
    rows = []
    order_id = 10001
    for d in dates:
        num_orders = np.random.randint(2, 6)
        for _ in range(num_orders):
            p = np.random.choice(products)
            qty = np.random.randint(1, 4)
            unit_price = p["price"]
            unit_cost = p["cost"]
            # Occasional promotional discount
            discount = 0.15 if np.random.random() < 0.15 else 0.0
            actual_price = round(unit_price * (1 - discount), 2)
            sales = round(actual_price * qty, 2)
            total_cost = round(unit_cost * qty, 2)
            profit = round(sales - total_cost, 2)
            
            rows.append({
                "Order_ID": f"ORD-{order_id}",
                "Order_Date": d.strftime("%Y-%m-%d"),
                "Product_Name": p["name"],
                "Category": p["category"],
                "Customer_ID": f"CUST-{np.random.randint(100, 350)}",
                "Quantity": qty,
                "Unit_Price": actual_price,
                "Sales_Revenue": sales,
                "COGS": total_cost,
                "Net_Profit": profit
            })
            order_id += 1
            
    return pd.DataFrame(rows)


# ==============================================================================
# Header
# ==============================================================================
st.title("💼 Business Performance Dashboard")
st.caption("Automated executive analytics, financial KPI monitoring, product profitability, and strategic directives.")

# Retrieve dataset
active_df: Optional[pd.DataFrame] = None
data_source_label = ""

if "Cleaned_Dataset" in st.session_state:
    active_df = st.session_state["Cleaned_Dataset"]
    data_source_label = "Cleaned Dataset"
elif "Dataset" in st.session_state:
    active_df = st.session_state["Dataset"]
    data_source_label = "Raw Uploaded Dataset (Uncleaned)"

if active_df is None or active_df.empty:
    st.info(
        "👋 **No dataset is currently loaded.** "
        "Please upload a dataset on the **📂 Data Processing** page to explore the dashboard."
    )
    st.stop()




# ==============================================================================
# Automatic Column Detection & User Override Drawer
# ==============================================================================
# Initialize or retrieve detected mapping in session state
if "detected_mapping" not in st.session_state or st.session_state.get("_last_df_shape") != active_df.shape:
    detected = detect_column_mapping(active_df)
    st.session_state["detected_mapping"] = detected
    st.session_state["_last_df_shape"] = active_df.shape

current_mapping: ColumnMapping = st.session_state["detected_mapping"]

with st.expander("🛠️ Dataset Schema & Column Mappings (Review or Override)", expanded=False):
    st.write(
        "DataInsight AI automatically detected the following column roles. "
        "You can adjust these mappings if your business uses specific field names."
    )
    
    col_options = ["None"] + list(active_df.columns)
    
    def get_index(col_name: Optional[str]) -> int:
        if col_name and col_name in col_options:
            return col_options.index(col_name)
        return 0

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    with m_col1:
        date_sel = st.selectbox("Date / Timestamp", col_options, index=get_index(current_mapping.date_col))
        prod_sel = st.selectbox("Product / SKU", col_options, index=get_index(current_mapping.product_col))
        cat_sel = st.selectbox("Category / Department", col_options, index=get_index(current_mapping.category_col))

    with m_col2:
        rev_sel = st.selectbox("Sales / Revenue", col_options, index=get_index(current_mapping.revenue_col))
        cost_sel = st.selectbox("Cost / COGS / Expenses", col_options, index=get_index(current_mapping.cost_col))
        profit_sel = st.selectbox("Net Profit / Income", col_options, index=get_index(current_mapping.profit_col))

    with m_col3:
        qty_sel = st.selectbox("Quantity / Units", col_options, index=get_index(current_mapping.quantity_col))
        price_sel = st.selectbox("Unit Price", col_options, index=get_index(current_mapping.price_col))
        order_sel = st.selectbox("Order / Transaction ID", col_options, index=get_index(current_mapping.order_id_col))

    with m_col4:
        cust_sel = st.selectbox("Customer / Account", col_options, index=get_index(current_mapping.customer_col))
        profit_mode_sel = st.selectbox(
            "Profit Calculation Method",
            ["Auto-detect", "Use Direct Profit Column", "Calculate (Revenue − Cost)"],
            index=0 if current_mapping.profit_mode == "auto" else (1 if current_mapping.profit_mode == "direct" else 2)
        )
        currency_choice = st.selectbox("Display Currency", ["$ (USD/CAD)", "€ (EUR)", "£ (GBP)", "₹ (INR)", "¥ (JPY)", "A$ (AUD)", "Neutral Numeric"], index=0)

    # Update mapping
    currency_symbol = currency_choice.split(" ")[0] if " " in currency_choice else "$"
    if currency_choice == "Neutral Numeric":
        currency_symbol = ""

    current_mapping.date_col = None if date_sel == "None" else date_sel
    current_mapping.product_col = None if prod_sel == "None" else prod_sel
    current_mapping.category_col = None if cat_sel == "None" else cat_sel
    current_mapping.revenue_col = None if rev_sel == "None" else rev_sel
    current_mapping.cost_col = None if cost_sel == "None" else cost_sel
    current_mapping.profit_col = None if profit_sel == "None" else profit_sel
    current_mapping.quantity_col = None if qty_sel == "None" else qty_sel
    current_mapping.price_col = None if price_sel == "None" else price_sel
    current_mapping.order_id_col = None if order_sel == "None" else order_sel
    current_mapping.customer_col = None if cust_sel == "None" else cust_sel

    if profit_mode_sel == "Use Direct Profit Column":
        current_mapping.profit_mode = "direct"
    elif profit_mode_sel == "Calculate (Revenue − Cost)":
        current_mapping.profit_mode = "calculated"
    else:
        current_mapping.profit_mode = "auto"


# ==============================================================================
# Filter Toolbar
# ==============================================================================
filtered_df = active_df.copy()

with st.container(border=True):
    f_col1, f_col2, f_col3 = st.columns([2, 2, 1])

    # Date Range Filter
    with f_col1:
        if current_mapping.date_col and current_mapping.date_col in filtered_df.columns:
            parsed_dates = pd.to_datetime(filtered_df[current_mapping.date_col], errors="coerce")
            valid_dates = parsed_dates.dropna()
            if not valid_dates.empty:
                min_d = valid_dates.min().date()
                max_d = valid_dates.max().date()
                date_range = st.date_input(
                    "📅 Filter Date Range",
                    value=(min_d, max_d),
                    min_value=min_d,
                    max_value=max_d
                )
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_d, end_d = date_range
                    mask = (parsed_dates.dt.date >= start_d) & (parsed_dates.dt.date <= end_d)
                    filtered_df = filtered_df[mask.fillna(False)]
            else:
                st.caption("📅 Date column mapped, but no valid dates found.")
        else:
            st.caption("📅 No date column mapped.")

    # Category Filter
    with f_col2:
        if current_mapping.category_col and current_mapping.category_col in filtered_df.columns:
            all_cats = sorted([str(c) for c in filtered_df[current_mapping.category_col].dropna().unique()])
            selected_cats = st.multiselect("📁 Filter Categories", options=all_cats, default=[])
            if selected_cats:
                filtered_df = filtered_df[filtered_df[current_mapping.category_col].astype(str).isin(selected_cats)]
        else:
            st.caption("📁 No category column mapped.")

    # Reset / Count Status
    with f_col3:
        st.markdown(f"**Data Status:** `{data_source_label}`")
        st.caption(f"Showing **{len(filtered_df):,}** of **{len(active_df):,}** records")


# ==============================================================================
# Core Business Calculations
# ==============================================================================
kpis = calculate_business_kpis(filtered_df, current_mapping)
product_perf_df = analyze_product_performance(filtered_df, current_mapping, kpis)
category_perf_df = analyze_category_performance(filtered_df, current_mapping)
recommendations = generate_business_recommendations(kpis, product_perf_df, category_perf_df, currency_symbol)
alert_result = detect_loss_making_products(filtered_df, current_mapping, currency_symbol)


# ==============================================================================
# Executive KPI Cards Grid
# ==============================================================================
st.markdown("### 📊 Executive KPI Overview")

if kpis.is_financial:
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        rev_val = format_currency(kpis.total_revenue, currency_symbol, decimals=0)
        rev_delta = f"{kpis.revenue_growth_pct:+.1f}% {kpis.growth_period_label or 'trend'}" if kpis.revenue_growth_pct is not None else None
        st.metric(label="Total Revenue", value=rev_val, delta=rev_delta)

    with kpi2:
        profit_val = format_currency(kpis.net_profit, currency_symbol, decimals=0)
        profit_delta = f"{kpis.profit_growth_pct:+.1f}% {kpis.growth_period_label or 'trend'}" if kpis.profit_growth_pct is not None else None
        st.metric(label="Net Profit", value=profit_val, delta=profit_delta)

    with kpi3:
        margin_val = format_percent(kpis.profit_margin_pct)
        margin_status = "Healthy" if (kpis.profit_margin_pct and kpis.profit_margin_pct >= 15) else "Review"
        st.metric(label="Profit Margin", value=margin_val, delta=margin_status if kpis.profit_margin_pct else None)

    with kpi4:
        loss_val = format_currency(kpis.total_loss_negative_records, currency_symbol, decimals=0)
        neg_count_str = f"{kpis.negative_records_count:,} negative orders" if kpis.negative_records_count else "Zero loss records"
        st.metric(label="Loss from Negative Records", value=loss_val, delta=f"-{neg_count_str}" if kpis.negative_records_count else "Optimal", delta_color="inverse")

    with kpi5:
        aov_val = format_currency(kpis.average_order_value, currency_symbol, decimals=2)
        aov_label = "Avg Order Value (AOV)" if kpis.is_aov_order_level else "Avg Rev / Transaction"
        vol_str = f"{kpis.unique_orders or kpis.transaction_count:,} orders"
        st.metric(label=aov_label, value=aov_val, delta=vol_str, delta_color="off")

else:
    # Non-financial dataset fallback KPI row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="Total Records", value=f"{kpis.transaction_count:,}")
    with kpi2:
        st.metric(label="Total Columns", value=f"{filtered_df.shape[1]}")
    with kpi3:
        st.metric(label="Unique Categories", value=f"{kpis.unique_categories or 'N/A'}")
    with kpi4:
        st.metric(label="Unique Products / Entities", value=f"{kpis.unique_products or 'N/A'}")

# KPI Notes and Explanations
with st.expander("ℹ️ Calculation Methodology, Definitions & Data Quality Notes", expanded=False):
    st.write(f"**Profit Source Definition:** {kpis.profit_source_explanation}")
    if kpis.notes_and_assumptions:
        st.markdown("**Assumptions & Notes:**")
        for note in kpis.notes_and_assumptions:
            st.markdown(f"- {note}")
    if kpis.quality_warnings:
        st.markdown("**Data Quality Warnings:**")
        for warn in kpis.quality_warnings:
            st.warning(warn)


# ==============================================================================
# Automatic AI Executive Briefing
# ==============================================================================
st.markdown("### 🧠 AI Executive Briefing")

groq_key = st.secrets.get("GROQ_API_KEY", None)

with st.container(border=True):
    with st.spinner("Synthesizing executive briefing..."):
        ai_briefing = generate_executive_ai_briefing(
            kpis=kpis,
            recommendations=recommendations,
            product_df=product_perf_df,
            category_df=category_perf_df,
            alerts=alert_result.alerts if alert_result.is_applicable else None,
            api_key=groq_key,
            currency_symbol=currency_symbol
        )
    st.markdown(ai_briefing)


# ==============================================================================
# Evidence-Based Business Risk Alerts
# ==============================================================================
st.markdown("### 🚨 Evidence-Based Business Risk Alerts")
st.caption("Automated financial alert system detecting loss-making products, calculating verified financial deficits, and providing audit-ready evidence.")

if not alert_result.is_applicable:
    st.info(f"ℹ️ **Business Alerts Notice:** {alert_result.status_message}")
elif alert_result.total_flagged_count == 0:
    st.success(f"✅ **No Loss-Making Products Detected:** {alert_result.status_message}")
else:
    # Summary Metric Cards
    al_col1, al_col2, al_col3, al_col4 = st.columns(4)
    with al_col1:
        st.metric(
            label="⚠️ Flagged Loss-Makers",
            value=f"{alert_result.total_flagged_count} SKU{'s' if alert_result.total_flagged_count > 1 else ''}",
            delta="Eroding Net Profits",
            delta_color="inverse"
        )
    with al_col2:
        comb_loss_str = format_currency(alert_result.total_combined_loss, currency_symbol, decimals=2)
        st.metric(
            label="🔻 Combined Net Deficit",
            value=comb_loss_str,
            delta="Total Capital Drag",
            delta_color="inverse"
        )
    with al_col3:
        top_loss_alert = alert_result.alerts[0]
        st.metric(
            label="🔥 Worst Loss Contributor",
            value=top_loss_alert.product_name,
            delta=f"-{format_currency(top_loss_alert.loss_amount, currency_symbol, decimals=2)}",
            delta_color="inverse"
        )
    with al_col4:
        loss_share_rev = (alert_result.total_combined_loss / kpis.total_revenue * 100) if (kpis.total_revenue and kpis.total_revenue > 0) else None
        st.metric(
            label="📉 Loss Share of Sales",
            value=format_percent(loss_share_rev) if loss_share_rev is not None else "N/A",
            delta="Of Gross Revenue",
            delta_color="off"
        )

    # Evidence Table
    st.markdown("#### 📋 Evidence Summary Table")
    if alert_result.evidence_df is not None and not alert_result.evidence_df.empty:
        display_alerts_df = alert_result.evidence_df.copy()
        display_alerts_df["Revenue"] = display_alerts_df["Revenue"].apply(lambda v: format_currency(v, currency_symbol, decimals=2))
        display_alerts_df["Cost"] = display_alerts_df["Cost"].apply(lambda v: format_currency(v, currency_symbol, decimals=2))
        display_alerts_df["Net Loss"] = display_alerts_df["Net Loss"].apply(lambda v: format_currency(v, currency_symbol, decimals=2))
        display_alerts_df["Profit Margin (%)"] = display_alerts_df["Profit Margin (%)"].apply(lambda v: format_percent(v, decimals=1))
        if "Share of Total Losses (%)" in display_alerts_df.columns:
            display_alerts_df["Share of Total Losses (%)"] = display_alerts_df["Share of Total Losses (%)"].apply(lambda v: format_percent(v, decimals=1))
        
        t_col_tab, t_col_btn = st.columns([4, 1])
        with t_col_tab:
            st.dataframe(display_alerts_df, use_container_width=True, hide_index=True)
        with t_col_btn:
            csv_alerts = alert_result.evidence_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Evidence CSV",
                data=csv_alerts,
                file_name="loss_making_products_evidence.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Detailed Expandable Alert Cards
    st.markdown("#### 🔍 Alert Investigations & Recommended Actions")
    for idx, alert in enumerate(alert_result.alerts, 1):
        margin_disp = format_percent(alert.profit_margin_pct) if alert.profit_margin_pct is not None else "N/A (Zero Rev)"
        with st.expander(
            f"🚨 **Alert #{idx}: {alert.product_name}** — Net Loss: **-{format_currency(alert.loss_amount, currency_symbol, decimals=2)}** | Margin: **{margin_disp}**",
            expanded=(idx == 1)
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Recorded Revenue", format_currency(alert.revenue, currency_symbol, decimals=2))
            c2.metric("Recorded Cost", format_currency(alert.cost, currency_symbol, decimals=2))
            c3.metric("Net Profit / Loss", format_currency(alert.net_profit, currency_symbol, decimals=2), delta_color="inverse")
            c4.metric("Profit Margin", margin_disp)

            st.markdown(f"**Explanation:** {alert.explanation}")
            st.markdown(f"**Suggested Action:** {alert.suggested_action}")

            if alert.notes:
                for note in alert.notes:
                    st.caption(f"ℹ️ {note}")

    # Optional AI Strategic Root-Cause Diagnostics
    with st.expander("🤖 AI Strategic Diagnostic & Investigation Directives (Advisory Only)", expanded=False):
        st.caption("AI-generated operational investigation hypotheses based strictly on verified pandas figures.")
        ai_alert_briefing = generate_alert_investigation_briefing(
            alerts=alert_result.alerts,
            api_key=groq_key,
            currency_symbol=currency_symbol
        )
        st.markdown(ai_alert_briefing)


# ==============================================================================
# Automatic Visual Analytics Section (2 to 4 Context-Aware Charts)
# ==============================================================================
st.markdown("### 📈 Visual Analytics & Business Dynamics")

# Layout: Two columns of visual cards
chart_col1, chart_col2 = st.columns(2)

# Chart 1: Trend over time OR Product Profitability
with chart_col1:
    trend_result = create_kpi_trend_chart(filtered_df, current_mapping, currency_symbol)
    if trend_result is not None:
        fig_trend, exp_trend = trend_result
        st.plotly_chart(fig_trend, use_container_width=True)
        st.info(f"💡 **Chart Insight:** {exp_trend}")
    elif product_perf_df is not None:
        profit_bar_res = create_product_profitability_chart(product_perf_df, currency_symbol)
        if profit_bar_res is not None:
            fig_bar, exp_bar = profit_bar_res
            st.plotly_chart(fig_bar, use_container_width=True)
            st.info(f"💡 **Chart Insight:** {exp_bar}")
    elif current_mapping.category_col:
        cat_dist_res = create_categorical_distribution_chart(filtered_df, current_mapping.category_col)
        if cat_dist_res is not None:
            fig_cat, exp_cat = cat_dist_res
            st.plotly_chart(fig_cat, use_container_width=True)
            st.info(f"💡 **Chart Insight:** {exp_cat}")
    else:
        num_cols = filtered_df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            num_dist_res = create_numeric_distribution_chart(filtered_df, num_cols[0])
            if num_dist_res:
                fig_num, exp_num = num_dist_res
                st.plotly_chart(fig_num, use_container_width=True)
                st.info(f"💡 **Chart Insight:** {exp_num}")

# Chart 2: Revenue vs Profit Scatter OR Category Performance
with chart_col2:
    scatter_res = create_revenue_vs_profit_chart(product_perf_df, currency_symbol) if product_perf_df is not None else None
    if scatter_res is not None:
        fig_scatter, exp_scatter = scatter_res
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.info(f"💡 **Chart Insight:** {exp_scatter}")
    elif category_perf_df is not None:
        cat_chart_res = create_category_performance_chart(category_perf_df, currency_symbol)
        if cat_chart_res is not None:
            fig_cat, exp_cat = cat_chart_res
            st.plotly_chart(fig_cat, use_container_width=True)
            st.info(f"💡 **Chart Insight:** {exp_cat}")
    else:
        num_cols = filtered_df.select_dtypes(include="number").columns
        if len(num_cols) > 1:
            num_dist_res = create_numeric_distribution_chart(filtered_df, num_cols[1])
            if num_dist_res:
                fig_num, exp_num = num_dist_res
                st.plotly_chart(fig_num, use_container_width=True)
                st.info(f"💡 **Chart Insight:** {exp_num}")


# ==============================================================================
# Product & Category Profitability Matrix
# ==============================================================================
if product_perf_df is not None and not product_perf_df.empty:
    st.markdown("### 🏆 Product Performance & Unit Economics")
    
    t_filter_col1, t_filter_col2, t_download_col = st.columns([2, 2, 1])
    
    with t_filter_col1:
        status_options = ["All Products"] + sorted(list(product_perf_df["Status"].unique()))
        selected_status = st.selectbox("Filter by Profitability Status", status_options, index=0)
    
    with t_filter_col2:
        search_query = st.text_input("🔍 Search Product Name", placeholder="e.g. Wireless")
        
    display_prod_df = product_perf_df.copy()
    if selected_status != "All Products":
        display_prod_df = display_prod_df[display_prod_df["Status"] == selected_status]
    if search_query.strip():
        display_prod_df = display_prod_df[display_prod_df["Product"].str.contains(search_query.strip(), case=False, na=False)]

    with t_download_col:
        st.write("")
        csv_data = display_prod_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="product_profitability_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Format table for visual presentation
    formatted_table = display_prod_df.copy()
    formatted_table["Revenue"] = formatted_table["Revenue"].apply(lambda v: format_currency(v, currency_symbol, decimals=2))
    formatted_table["Cost"] = formatted_table["Cost"].apply(lambda v: format_currency(v, currency_symbol, decimals=2))
    formatted_table["Net Profit"] = formatted_table["Net Profit"].apply(lambda v: format_currency(v, currency_symbol, decimals=2))
    formatted_table["Profit Margin (%)"] = formatted_table["Profit Margin (%)"].apply(lambda v: format_percent(v, decimals=1))
    formatted_table["Loss from Negatives"] = formatted_table["Loss from Negatives"].apply(lambda v: format_currency(v, currency_symbol, decimals=2))
    formatted_table["Transactions"] = formatted_table["Transactions"].apply(lambda v: f"{v:,}")

    st.dataframe(formatted_table, use_container_width=True, hide_index=True)


# ==============================================================================
# Transparent, Evidence-Based Recommendations
# ==============================================================================
st.markdown("### 🎯 Strategic Business Recommendations")

if recommendations:
    for rec in recommendations:
        priority_class = "rec-card"
        if "High" in rec.priority:
            priority_class += " rec-high"
        elif "Opportunity" in rec.priority:
            priority_class += " rec-opp"
        elif "Medium" in rec.priority:
            priority_class += " rec-med"

        st.markdown(f"""
        <div class="{priority_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: 700; font-size: 1.05rem; color: #0F172A;">{rec.title}</span>
                <span style="font-size: 0.8rem; font-weight: 600; padding: 2px 8px; border-radius: 4px; background: #E2E8F0; color: #334155;">{rec.priority} • {rec.category}</span>
            </div>
            <div style="font-size: 0.92rem; color: #334155; margin-bottom: 6px;"><b>Observation:</b> {rec.observation}</div>
            <div style="font-size: 0.92rem; color: #334155; margin-bottom: 6px;"><b>Supporting Evidence:</b> {rec.evidence}</div>
            <div style="font-size: 0.92rem; color: #475569; margin-bottom: 8px;"><b>Business Impact:</b> {rec.business_impact}</div>
            <div style="font-size: 0.92rem; color: #1E40AF; background: #EFF6FF; padding: 8px 12px; border-radius: 4px;"><b>Recommended Investigation:</b> {rec.next_steps}</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No actionable margin anomalies or concentration risks detected in the active filtered view.")


# ==============================================================================
# Advanced Manual Visualization Studio (Expander for Power Users)
# ==============================================================================
with st.expander("🛠️ Advanced Visualization Studio (Analyst Sandbox)", expanded=False):
    st.markdown("##### Custom Visualization & Ad-Hoc Analytics")
    st.caption("Configure custom chart axes, measure aggregations, color groupings, and dimensions.")

    s_col1, s_col2, s_col3 = st.columns(3)
    
    with s_col1:
        studio_chart_type = st.selectbox(
            "Chart Type",
            ["Bar Chart", "Horizontal Bar", "Line Chart", "Area Chart", "Scatter Plot", "Histogram", "Box Plot", "Donut / Pie Chart", "Correlation Heatmap"],
            key="studio_type"
        )
        x_axis_col = st.selectbox("X-Axis / Category Column", list(filtered_df.columns), key="studio_x")

    with s_col2:
        num_options = ["None"] + [c for c in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[c])]
        y_axis_col = st.selectbox("Y-Axis / Measure Column", num_options, key="studio_y")
        agg_choice = st.selectbox("Aggregation Method", ["Sum", "Mean", "Median", "Count", "Min", "Max"], index=0, key="studio_agg")

    with s_col3:
        color_options = ["None"] + [c for c in filtered_df.columns if filtered_df[c].nunique() <= 30]
        color_group_col = st.selectbox("Color / Group By (Optional)", color_options, key="studio_color")
        custom_chart_title = st.text_input("Custom Chart Title", placeholder="e.g. Sales by Region & Segment", key="studio_title")

    if st.button("🚀 Render Custom Chart", use_container_width=True):
        try:
            studio_fig = create_custom_studio_chart(
                df=filtered_df,
                chart_type=studio_chart_type,
                x_col=x_axis_col,
                y_col=y_axis_col,
                color_col=color_group_col,
                agg_func=agg_choice,
                title=custom_chart_title
            )
            st.plotly_chart(studio_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not render custom chart: {e}")


# ==============================================================================
# Exportable Executive Summary Document
# ==============================================================================
st.divider()
st.markdown("### 📄 Export Executive Briefing Report")

export_col1, export_col2 = st.columns([3, 1])

with export_col1:
    st.caption("Generate a documented text summary of the current executive dashboard findings and recommendations for distribution.")

with export_col2:
    report_text = f"""====================================================
DATAINSIGHT AI — EXECUTIVE PERFORMANCE REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Data Source: {data_source_label} ({len(filtered_df):,} records analyzed)
====================================================

1. EXECUTIVE BUSINESS METRICS
----------------------------------------------------
Total Revenue               : {format_currency(kpis.total_revenue, currency_symbol)}
Total Cost                  : {format_currency(kpis.total_cost, currency_symbol)}
Net Profit                  : {format_currency(kpis.net_profit, currency_symbol)}
Profit Margin               : {format_percent(kpis.profit_margin_pct)}
Loss from Negative Records  : {format_currency(kpis.total_loss_negative_records, currency_symbol)} ({kpis.negative_records_count} records)
Average Order Value         : {format_currency(kpis.average_order_value, currency_symbol)}
Period Growth Rate          : {f'{kpis.revenue_growth_pct:+.1f}%' if kpis.revenue_growth_pct is not None else 'N/A'}
Profit Definition           : {kpis.profit_source_explanation}

2. AI EXECUTIVE BRIEFING
----------------------------------------------------
{ai_briefing}

3. STRATEGIC RECOMMENDATIONS
----------------------------------------------------
"""
    for idx, r in enumerate(recommendations, 1):
        report_text += f"\n[{idx}] {r.title} ({r.priority} • {r.category})\n"
        report_text += f"    - Observation : {r.observation}\n"
        report_text += f"    - Evidence    : {r.evidence}\n"
        report_text += f"    - Impact      : {r.business_impact}\n"
        report_text += f"    - Action      : {r.next_steps}\n"

    if alert_result.is_applicable:
        report_text += f"\n4. EVIDENCE-BASED BUSINESS RISK ALERTS\n----------------------------------------------------\n"
        report_text += f"Status: {alert_result.status_message}\n"
        report_text += f"Total Flagged Products: {alert_result.total_flagged_count}\n"
        report_text += f"Combined Net Loss: {format_currency(alert_result.total_combined_loss, currency_symbol)}\n"
        for idx, a in enumerate(alert_result.alerts, 1):
            margin_str = format_percent(a.profit_margin_pct) if a.profit_margin_pct is not None else "N/A"
            report_text += f"\n[{idx}] {a.product_name} | Net Loss: {format_currency(a.loss_amount, currency_symbol)} | Margin: {margin_str}\n"
            report_text += f"    - Revenue: {format_currency(a.revenue, currency_symbol)} | Cost: {format_currency(a.cost, currency_symbol)}\n"
            report_text += f"    - Explanation: {a.explanation}\n"
            report_text += f"    - Action     : {a.suggested_action}\n"

    report_text += "\n====================================================\nGenerated with DataInsight AI\n"

    st.download_button(
        label="📄 Download Executive Report (.txt)",
        data=report_text.encode("utf-8"),
        file_name=f"DataInsight_Executive_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True
    )
