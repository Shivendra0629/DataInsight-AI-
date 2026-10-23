"""
AI Business Insights Integration Engine for DataInsight AI.
Connects with Groq LLM to generate executive business briefings based strictly on
verified pandas-calculated KPIs, product profitability rankings, and risk indicators.
Guarantees deterministic fallback when the API is unconfigured or offline.
"""

from typing import List, Optional
import os
import pandas as pd
from groq import Groq
from utils.analytics import BusinessKPIs, Recommendation, BusinessAlert
from utils.helper import format_currency, format_percent


def generate_executive_ai_briefing(
    kpis: BusinessKPIs,
    recommendations: List[Recommendation],
    product_df: Optional[pd.DataFrame] = None,
    category_df: Optional[pd.DataFrame] = None,
    alerts: Optional[List[BusinessAlert]] = None,
    api_key: Optional[str] = None,
    currency_symbol: str = "$",
    model: str = "llama-3.3-70b-versatile"
) -> str:
    """
    Generates a concise, high-level executive briefing.
    Uses verified calculated metrics and transparent evidence.
    Falls back to a deterministic executive synthesis if LLM is unavailable.
    """
    # Deterministic synthesis generator
    def build_deterministic_summary() -> str:
        lines = []
        if kpis.is_financial:
            rev_str = format_currency(kpis.total_revenue, currency_symbol, decimals=0)
            profit_str = format_currency(kpis.net_profit, currency_symbol, decimals=0)
            margin_str = format_percent(kpis.profit_margin_pct)
            
            lines.append(f"### 💼 Executive Financial Briefing")
            lines.append(
                f"During the analyzed operating period, the business generated **{rev_str}** in gross revenue, "
                f"resulting in a cumulative net profit of **{profit_str}** (an overall profit margin of **{margin_str}**)."
            )
            
            if alerts and len(alerts) > 0:
                total_loss_alert = sum(a.loss_amount for a in alerts)
                lines.append(
                    f"\n🚨 **Loss-Making Product Risk:** Identified **{len(alerts)} loss-making SKU(s)** incurring "
                    f"a verified cumulative loss of **{format_currency(total_loss_alert, currency_symbol, decimals=0)}** requiring operational review."
                )
            elif kpis.total_loss_negative_records and kpis.total_loss_negative_records > 0:
                loss_str = format_currency(kpis.total_loss_negative_records, currency_symbol, decimals=0)
                lines.append(
                    f"\n⚠️ **Profitability Leakage Detected:** {kpis.negative_records_count} transaction record(s) incurred net losses "
                    f"totaling **{loss_str}**, eroding top-line earnings."
                )

            if product_df is not None and not product_df.empty:
                top_p = product_df[product_df["Net Profit"] > 0].head(1)
                loss_p = product_df[product_df["Net Profit"] < 0].head(1)
                if not top_p.empty:
                    top_name = top_p.iloc[0]["Product"]
                    top_profit = format_currency(top_p.iloc[0]["Net Profit"], currency_symbol, decimals=0)
                    lines.append(f"\n🌟 **Primary Profit Driver:** '{top_name}' generated **{top_profit}** in net profit.")
                if not loss_p.empty and not alerts:
                    loss_name = loss_p.iloc[0]["Product"]
                    loss_amt = format_currency(abs(loss_p.iloc[0]["Net Profit"]), currency_symbol, decimals=0)
                    lines.append(f"🔻 **Immediate Margin Risk:** '{loss_name}' posted a cumulative net loss of **{loss_amt}**.")

            if kpis.revenue_growth_pct is not None:
                growth_sign = "+" if kpis.revenue_growth_pct > 0 else ""
                lines.append(
                    f"\n📈 **Growth Velocity:** {kpis.growth_period_label or 'Period'} revenue shifted by **{growth_sign}{kpis.revenue_growth_pct:.1f}%**."
                )
        else:
            lines.append(f"### 📋 General Operational Summary")
            lines.append(
                f"Processed **{kpis.transaction_count:,} total records** across available dimensions. "
                "No direct financial fields (revenue/profit) were mapped, so analysis is focused on volume, distributions, and category segmentation."
            )

        if recommendations:
            lines.append("\n#### 🎯 Strategic Priority Action Items:")
            for i, rec in enumerate(recommendations[:3], 1):
                lines.append(f"{i}. **{rec.title}** — {rec.observation} *({rec.next_steps})*")

        return "\n".join(lines)

    # If no API key provided, return deterministic summary immediately
    if not api_key or not api_key.strip():
        return build_deterministic_summary()

    # Prepare compact, verified numerical payload for Groq LLM
    kpi_payload = {
        "Total Revenue": format_currency(kpis.total_revenue, currency_symbol),
        "Total Cost": format_currency(kpis.total_cost, currency_symbol),
        "Net Profit": format_currency(kpis.net_profit, currency_symbol),
        "Profit Margin": format_percent(kpis.profit_margin_pct),
        "Loss from Negative Records": format_currency(kpis.total_loss_negative_records, currency_symbol),
        "Negative Records Count": kpis.negative_records_count,
        "Total Transactions": kpis.transaction_count,
        "Unique Orders": kpis.unique_orders,
        "Unique Products": kpis.unique_products,
        "Average Order Value": format_currency(kpis.average_order_value, currency_symbol),
        "Revenue Growth Rate": f"{kpis.revenue_growth_pct:+.1f}%" if kpis.revenue_growth_pct is not None else "N/A",
        "Profit Growth Rate": f"{kpis.profit_growth_pct:+.1f}%" if kpis.profit_growth_pct is not None else "N/A",
        "Profit Definition Used": kpis.profit_source_explanation
    }

    if alerts:
        kpi_payload["Loss-Making Products Flagged"] = len(alerts)
        kpi_payload["Combined Flagged Loss"] = format_currency(sum(a.loss_amount for a in alerts), currency_symbol)

    product_summary = []
    if product_df is not None and not product_df.empty:
        top_prods = product_df.head(3)[["Product", "Revenue", "Net Profit", "Profit Margin (%)", "Status"]].to_dict(orient="records")
        product_summary = top_prods

    recs_summary = [
        {"Title": r.title, "Observation": r.observation, "Impact": r.business_impact, "NextSteps": r.next_steps}
        for r in recommendations[:4]
    ]

    alert_summary = []
    if alerts:
        alert_summary = [
            {"Product": a.product_name, "Revenue": format_currency(a.revenue, currency_symbol), "Cost": format_currency(a.cost, currency_symbol), "Loss": format_currency(a.loss_amount, currency_symbol), "Margin": format_percent(a.profit_margin_pct)}
            for a in alerts[:4]
        ]

    system_prompt = (
        "You are an executive SaaS Business Intelligence consultant. "
        "Your task is to provide an elite, concise 3-paragraph executive performance briefing based EXCLUSIVELY on "
        "the pre-calculated metrics supplied below. "
        "STRICT INSTRUCTIONS: "
        "- Do NOT invent, assume, or hallucinate any numbers or dates. "
        "- Use the exact figures provided. "
        "- Structure with 3 clear sections: 'Executive Overview', 'Margin & Risk Analysis', and 'Immediate Strategic Directives'. "
        "- Maintain a crisp, professional C-suite tone."
    )

    user_prompt = f"""
Calculated Business KPIs:
{kpi_payload}

Top Products Performance:
{product_summary}

Loss-Making Products Detected:
{alert_summary}

Rule-Based Recommendations Identified:
{recs_summary}

Generate the executive briefing now.
"""

    try:
        client = Groq(api_key=api_key)
        target_model = model if model else "llama-3.3-70b-versatile"
        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=600
        )
        content = response.choices[0].message.content
        if content and len(content.strip()) > 50:
            return content.strip()
        return build_deterministic_summary()
    except Exception:
        # Graceful fallback to verified deterministic summary without crashing
        return build_deterministic_summary()


def generate_alert_investigation_briefing(
    alerts: List[BusinessAlert],
    api_key: Optional[str] = None,
    currency_symbol: str = "$",
    model: str = "llama-3.3-70b-versatile"
) -> str:
    """
    Generates an executive diagnostic investigation briefing for flagged loss-making products.
    Uses Groq LLM if configured; strictly relies on verified pandas figures.
    Falls back to a structured deterministic root-cause investigation playbook when offline.
    """
    if not alerts:
        return "No active loss-making product alerts requiring investigation."

    total_loss = sum(a.loss_amount for a in alerts)
    worst_alert = alerts[0]

    def build_deterministic_investigation() -> str:
        lines = [
            f"#### 🔍 Operational Root-Cause Investigation Framework",
            f"**Verified Findings:** Identified **{len(alerts)} product(s)** operating at a cumulative net loss of **{format_currency(total_loss, currency_symbol)}**.",
            f"Primary loss driver: **'{worst_alert.product_name}'** accounting for **{format_currency(worst_alert.loss_amount, currency_symbol)}** in deficits.",
            "",
            "**Recommended Diagnostic Audit Steps:**",
            "1. **Supplier Cost Audit (COGS):** Check recent supplier invoices for raw material price increases, freight surcharges, or minimum order fee penalties.",
            "2. **Promotional Discount Review:** Audit applied discount codes or promotional pricing rules to confirm whether sales were closed below unit cost.",
            "3. **Return & Refund Diagnostics:** Inspect transaction logs for high return rates, return shipping absorption, or restocking fee waivers on these SKUs.",
            "4. **Strategic Role Assessment:** Verify whether the item is an intentional 'loss-leader' designed for customer acquisition before altering pricing or phasing out."
        ]
        return "\n".join(lines)

    if not api_key or not api_key.strip():
        return build_deterministic_investigation()

    # If API key is available, construct safe prompt with verified numbers
    alert_payload = [
        {
            "Product": a.product_name,
            "Category": a.category or "N/A",
            "Revenue": format_currency(a.revenue, currency_symbol),
            "Cost": format_currency(a.cost, currency_symbol),
            "Net Loss": format_currency(a.loss_amount, currency_symbol),
            "Profit Margin": format_percent(a.profit_margin_pct),
            "Transactions": a.transactions
        }
        for a in alerts[:5]
    ]

    system_prompt = (
        "You are an executive operational auditor and financial consultant. "
        "Analyze the following verified loss-making products detected by our analytics engine. "
        "STRICT CONSTRAINTS: "
        "- Do NOT recalculate or modify the financial figures provided. "
        "- Do NOT invent or fabricate any numbers, dates, or causes. "
        "- Provide 3 concrete operational investigation hypotheses (e.g. supplier COGS drift, discounting leakages, shipping/returns). "
        "- Clearly label your hypotheses as 'Advisory Hypotheses to Investigate'."
    )

    user_prompt = f"""
Verified Loss-Making Product Alerts:
Total Loss Across Flagged Products: {format_currency(total_loss, currency_symbol)}
Products Flagged: {len(alerts)}
Detailed Alert Data:
{alert_payload}

Generate a concise operational investigation directive now.
"""
    try:
        client = Groq(api_key=api_key)
        target_model = model if model else "llama-3.3-70b-versatile"
        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        content = response.choices[0].message.content
        if content and len(content.strip()) > 50:
            return content.strip()
        return build_deterministic_investigation()
    except Exception:
        return build_deterministic_investigation()
