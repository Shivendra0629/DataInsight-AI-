"""
Comprehensive Test Suite for DataInsight AI Business Performance Dashboard.
Validates all 10 edge cases stipulated in Section 12:
1. Dataset with product, revenue, and cost
2. Dataset with existing profit column
3. Dataset with dates and revenue
4. Dataset with categories but no financial columns
5. Dataset with missing values and duplicate records
6. Dataset with negative profits
7. Dataset with zero revenue
8. Dataset with misleading or ambiguous column names
9. An empty dataset
10. Dataset with many categories or products
"""

import unittest
import pandas as pd
import numpy as np

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
from utils.ai_insights import generate_alert_investigation_briefing
from utils.charts import (
    create_kpi_trend_chart,
    create_product_profitability_chart,
    create_revenue_vs_profit_chart,
    create_category_performance_chart,
    create_categorical_distribution_chart,
    create_numeric_distribution_chart,
    create_custom_studio_chart
)
from utils.ai_insights import generate_executive_ai_briefing
from utils.helper import format_currency, format_percent, format_number


class TestBusinessDashboardAnalytics(unittest.TestCase):

    def test_case_1_product_revenue_cost(self):
        """Case 1: Dataset containing product, revenue, and cost."""
        df = pd.DataFrame({
            "product_name": ["Widget A", "Widget B", "Widget C"],
            "total_revenue": [1000.0, 2000.0, 1500.0],
            "total_cost": [600.0, 1400.0, 900.0]
        })
        mapping = detect_column_mapping(df)
        self.assertEqual(mapping.product_col, "product_name")
        self.assertEqual(mapping.revenue_col, "total_revenue")
        self.assertEqual(mapping.cost_col, "total_cost")

        kpis = calculate_business_kpis(df, mapping)
        self.assertTrue(kpis.is_financial)
        self.assertEqual(kpis.total_revenue, 4500.0)
        self.assertEqual(kpis.total_cost, 2900.0)
        # Net Profit = 4500 - 2900 = 1600
        self.assertEqual(kpis.net_profit, 1600.0)
        # Profit Margin = 1600 / 4500 * 100 = 35.555...
        self.assertAlmostEqual(kpis.profit_margin_pct, (1600.0 / 4500.0) * 100, places=2)
        self.assertEqual(kpis.transaction_count, 3)

        # Product performance
        prod_df = analyze_product_performance(df, mapping, kpis)
        self.assertIsNotNone(prod_df)
        self.assertEqual(len(prod_df), 3)
        top_prod = prod_df.iloc[0]
        self.assertEqual(top_prod["Product"], "Widget B")
        self.assertEqual(top_prod["Net Profit"], 600.0)

    def test_case_2_existing_profit_column(self):
        """Case 2: Dataset containing an existing profit column."""
        df = pd.DataFrame({
            "item_name": ["Alpha", "Beta", "Gamma"],
            "sales": [5000.0, 3000.0, 2000.0],
            "net_profit": [1200.0, 800.0, 400.0]
        })
        mapping = detect_column_mapping(df)
        self.assertEqual(mapping.profit_col, "net_profit")
        self.assertEqual(mapping.profit_mode, "direct")

        kpis = calculate_business_kpis(df, mapping)
        self.assertTrue(kpis.is_financial)
        self.assertEqual(kpis.total_revenue, 10000.0)
        self.assertEqual(kpis.net_profit, 2400.0)
        self.assertAlmostEqual(kpis.profit_margin_pct, 24.0, places=2)
        self.assertIn("Direct profit column", kpis.profit_source_explanation)

    def test_case_3_dates_and_revenue(self):
        """Case 3: Dataset containing dates and revenue."""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        df = pd.DataFrame({
            "order_date": dates,
            "revenue": np.linspace(100, 500, 60)
        })
        mapping = detect_column_mapping(df)
        self.assertEqual(mapping.date_col, "order_date")
        self.assertEqual(mapping.revenue_col, "revenue")

        kpis = calculate_business_kpis(df, mapping)
        self.assertTrue(kpis.is_financial)
        self.assertEqual(kpis.transaction_count, 60)
        self.assertIsNotNone(kpis.growth_period_label)

        # Test trend chart generation
        trend_res = create_kpi_trend_chart(df, mapping)
        self.assertIsNotNone(trend_res)
        fig, exp = trend_res
        self.assertTrue(len(fig.data) >= 1)
        self.assertIn("trajectory", exp.lower())

    def test_case_4_categories_no_financials(self):
        """Case 4: Dataset containing categories but no financial columns."""
        df = pd.DataFrame({
            "category": ["Hardware", "Software", "Hardware", "Services", "Software", "Hardware"],
            "user_id": [101, 102, 103, 104, 105, 106]
        })
        mapping = detect_column_mapping(df)
        self.assertEqual(mapping.category_col, "category")
        self.assertIsNone(mapping.revenue_col)
        self.assertIsNone(mapping.cost_col)
        self.assertIsNone(mapping.profit_col)

        kpis = calculate_business_kpis(df, mapping)
        self.assertFalse(kpis.is_financial)
        self.assertIsNone(kpis.total_revenue)
        self.assertIsNone(kpis.net_profit)
        self.assertEqual(kpis.transaction_count, 6)
        self.assertEqual(kpis.unique_categories, 3)

        # Fallback distribution chart works without crash
        dist_res = create_categorical_distribution_chart(df, "category")
        self.assertIsNotNone(dist_res)
        fig, exp = dist_res
        self.assertIn("Hardware", list(fig.data[0].x))

        # Recommendations contain clear advisory about missing financials
        recs = generate_business_recommendations(kpis)
        self.assertTrue(any("Financial Fields" in r.title for r in recs))

    def test_case_5_missing_values_and_duplicates(self):
        """Case 5: Dataset with missing values and duplicate records."""
        df = pd.DataFrame({
            "product": ["Widget A", "Widget A", None, "Widget B", "Widget C"],
            "revenue": [100.0, 100.0, np.nan, "invalid_num", 300.0],
            "cost": [50.0, 50.0, 20.0, 40.0, None]
        })
        mapping = detect_column_mapping(df)
        kpis = calculate_business_kpis(df, mapping)
        # Should not crash, should handle NaN and invalid string smoothly
        self.assertIsNotNone(kpis.total_revenue)
        self.assertTrue(len(kpis.quality_warnings) > 0)
        self.assertEqual(kpis.transaction_count, 5)

    def test_case_6_negative_profits(self):
        """Case 6: Dataset with negative profits."""
        df = pd.DataFrame({
            "product": ["Profitable X", "Loss Maker Y", "Loss Maker Z"],
            "revenue": [1000.0, 500.0, 200.0],
            "cost": [600.0, 800.0, 350.0]  # Profits: +400, -300, -150
        })
        mapping = detect_column_mapping(df)
        kpis = calculate_business_kpis(df, mapping)
        self.assertEqual(kpis.total_revenue, 1700.0)
        self.assertEqual(kpis.total_cost, 1750.0)
        self.assertEqual(kpis.net_profit, -50.0)
        # Total loss from negative records = |-300| + |-150| = 450.0
        self.assertEqual(kpis.total_loss_negative_records, 450.0)
        self.assertEqual(kpis.negative_records_count, 2)

        prod_df = analyze_product_performance(df, mapping, kpis)
        self.assertIsNotNone(prod_df)
        loss_makers = prod_df[prod_df["Status"].str.contains("Loss-Maker")]
        self.assertEqual(len(loss_makers), 2)

        # Recommendations should catch negative profit products
        recs = generate_business_recommendations(kpis, prod_df)
        self.assertTrue(any("Unprofitable Products" in r.title for r in recs))

    def test_case_7_zero_revenue(self):
        """Case 7: Dataset with zero revenue."""
        df = pd.DataFrame({
            "product": ["Free Tier A", "Free Tier B"],
            "revenue": [0.0, 0.0],
            "cost": [10.0, 20.0]
        })
        mapping = detect_column_mapping(df)
        kpis = calculate_business_kpis(df, mapping)
        self.assertEqual(kpis.total_revenue, 0.0)
        self.assertEqual(kpis.net_profit, -30.0)
        # Margin should be handled safely without ZeroDivisionError
        self.assertEqual(kpis.profit_margin_pct, 0.0)

    def test_case_8_misleading_or_ambiguous_column_names(self):
        """Case 8: Dataset with misleading or ambiguous column names."""
        df = pd.DataFrame({
            "cost_center_code": ["CC1", "CC2", "CC3"],
            "order_total_sales_amount": [500.0, 1000.0, 750.0],
            "net_profit_margin_amount": [100.0, 250.0, 150.0],
            "shipping_days_estimate": [2, 5, 3]
        })
        mapping = detect_column_mapping(df)
        self.assertEqual(mapping.revenue_col, "order_total_sales_amount")
        self.assertEqual(mapping.profit_col, "net_profit_margin_amount")
        self.assertNotEqual(mapping.cost_col, "cost_center_code")  # Should not map non-numeric CC code as cost

        kpis = calculate_business_kpis(df, mapping)
        self.assertEqual(kpis.total_revenue, 2250.0)
        self.assertEqual(kpis.net_profit, 500.0)

    def test_case_9_empty_dataset(self):
        """Case 9: An empty dataset."""
        df = pd.DataFrame()
        mapping = detect_column_mapping(df)
        kpis = calculate_business_kpis(df, mapping)
        self.assertFalse(kpis.is_financial)
        self.assertEqual(kpis.transaction_count, 0)
        self.assertIsNone(kpis.total_revenue)
        self.assertIsNone(kpis.net_profit)

        prod_df = analyze_product_performance(df, mapping)
        self.assertIsNone(prod_df)

    def test_case_10_many_categories_or_products(self):
        """Case 10: Dataset with many categories or products (100+ unique items)."""
        np.random.seed(99)
        num_items = 200
        df = pd.DataFrame({
            "product_id": [f"SKU_{i:04d}" for i in range(num_items)],
            "category_name": [f"Category_{i % 25}" for i in range(num_items)],
            "sales_turnover": np.random.uniform(50, 5000, num_items),
            "cost_price": np.random.uniform(30, 4500, num_items)
        })
        mapping = detect_column_mapping(df)
        self.assertEqual(mapping.product_col, "product_id")
        self.assertEqual(mapping.category_col, "category_name")
        self.assertEqual(mapping.revenue_col, "sales_turnover")

        kpis = calculate_business_kpis(df, mapping)
        self.assertEqual(kpis.transaction_count, num_items)
        self.assertEqual(kpis.unique_products, num_items)
        self.assertEqual(kpis.unique_categories, 25)

        prod_df = analyze_product_performance(df, mapping, kpis)
        self.assertIsNotNone(prod_df)
        self.assertEqual(len(prod_df), num_items)

        # Charts render safely without choking on 200 items
        fig, exp = create_product_profitability_chart(prod_df)
        self.assertIsNotNone(fig)
        self.assertTrue(len(fig.data[0].y) <= 12)  # Limited to top/bottom cleanly


class TestFormattingAndStudio(unittest.TestCase):

    def test_format_helpers(self):
        self.assertEqual(format_currency(1234.56, "$"), "$1,234.56")
        self.assertEqual(format_currency(-500.0, "$"), "-$500.00")
        self.assertEqual(format_currency(1_500_000, "$", abbreviate=True), "$1.50M")
        self.assertEqual(format_currency(None, "$"), "N/A")
        self.assertEqual(format_percent(25.67), "25.7%")
        self.assertEqual(format_percent(None), "N/A")

    def test_studio_chart_generation(self):
        df = pd.DataFrame({
            "Region": ["East", "West", "North", "South"],
            "Sales": [100.0, 200.0, 150.0, 300.0],
            "Profit": [20.0, 45.0, 10.0, 80.0]
        })
        fig = create_custom_studio_chart(df, "Bar Chart", x_col="Region", y_col="Sales", agg_func="sum")
        self.assertIsNotNone(fig)
        self.assertEqual(fig.layout.title.text, "<b>Bar Chart: Region vs Sales</b>")




class TestEvidenceBasedBusinessAlerts(unittest.TestCase):
    """
    Validates all functional requirements for the Evidence-Based Business Alert System:
    1. Product with positive profit is not flagged.
    2. Product with negative profit is flagged correctly with full financial evidence.
    3. Product with zero profit is not flagged.
    4. Multiple loss-making products are aggregated correctly.
    5. Missing financial columns are handled safely.
    6. Empty dataset is handled safely.
    7. Missing and non-numeric values do not produce misleading results.
    8. Zero revenue does not cause division-by-zero errors in margin calculations.
    9. Existing filters and custom column mappings are respected.
    10. Existing analytics functions remain unbroken.
    11. Unit cost vs total cost compatibility.
    12. Deterministic offline investigation fallback without external API call.
    """

    def test_1_positive_profit_not_flagged(self):
        """1. A product with positive profit is not flagged."""
        df = pd.DataFrame({
            "product_name": ["Profitable Gadget A", "Profitable Gadget B"],
            "revenue": [1000.0, 2500.0],
            "cost": [600.0, 1800.0]  # Profits: +400.0, +700.0
        })
        mapping = detect_column_mapping(df)
        result = detect_loss_making_products(df, mapping)

        self.assertTrue(result.is_applicable)
        self.assertEqual(result.total_flagged_count, 0)
        self.assertEqual(len(result.alerts), 0)
        self.assertEqual(result.total_combined_loss, 0.0)
        self.assertIn("No loss-making products detected", result.status_message)

    def test_2_negative_profit_flagged_correctly(self):
        """2. A product with negative profit is flagged correctly with financial evidence."""
        df = pd.DataFrame({
            "product_name": ["Product A", "Product B"],
            "revenue": [8000.0, 5000.0],
            "cost": [10000.0, 3000.0]  # Product A: profit = -2000.0; Product B: profit = +2000.0
        })
        mapping = detect_column_mapping(df)
        result = detect_loss_making_products(df, mapping, currency_symbol="₹")

        self.assertTrue(result.is_applicable)
        self.assertEqual(result.total_flagged_count, 1)
        self.assertEqual(len(result.alerts), 1)

        alert = result.alerts[0]
        self.assertEqual(alert.product_name, "Product A")
        self.assertEqual(alert.revenue, 8000.0)
        self.assertEqual(alert.cost, 10000.0)
        self.assertEqual(alert.net_profit, -2000.0)
        self.assertEqual(alert.loss_amount, 2000.0)
        # Margin = -2000 / 8000 * 100 = -25.0%
        self.assertAlmostEqual(alert.profit_margin_pct, -25.0, places=2)
        self.assertIn("Product A", alert.title)
        self.assertIn("exceeds", alert.explanation.lower())
        self.assertIn("₹10,000", alert.explanation)
        self.assertIn("₹8,000", alert.explanation)
        self.assertIn("₹2,000", alert.explanation)
        self.assertIn("price", alert.suggested_action.lower())
        self.assertIn("cost", alert.suggested_action.lower())

    def test_3_zero_profit_not_flagged(self):
        """3. A product with zero profit is not flagged as loss-making (break-even)."""
        df = pd.DataFrame({
            "product_name": ["Break Even Widget", "Profitable Widget"],
            "revenue": [1500.0, 3000.0],
            "cost": [1500.0, 2000.0]  # Profit: 0.0, +1000.0
        })
        mapping = detect_column_mapping(df)
        result = detect_loss_making_products(df, mapping)

        self.assertTrue(result.is_applicable)
        self.assertEqual(result.total_flagged_count, 0)
        self.assertEqual(len(result.alerts), 0)
        self.assertEqual(result.total_combined_loss, 0.0)

    def test_4_multiple_loss_making_products_aggregated(self):
        """4. Multiple loss-making products are aggregated correctly."""
        df = pd.DataFrame({
            "product_name": ["Loss Maker X", "Loss Maker Y", "Winner Z"],
            "revenue": [2000.0, 1000.0, 5000.0],
            "cost": [3000.0, 1500.0, 4000.0]  # Profits: -1000.0, -500.0, +1000.0
        })
        mapping = detect_column_mapping(df)
        result = detect_loss_making_products(df, mapping, currency_symbol="$")

        self.assertTrue(result.is_applicable)
        self.assertEqual(result.total_flagged_count, 2)
        self.assertEqual(result.total_combined_loss, 1500.0)  # |-1000| + |-500|

        # Sorted descending by loss (worst loss first)
        self.assertEqual(result.alerts[0].product_name, "Loss Maker X")
        self.assertEqual(result.alerts[0].loss_amount, 1000.0)
        self.assertEqual(result.alerts[1].product_name, "Loss Maker Y")
        self.assertEqual(result.alerts[1].loss_amount, 500.0)

        # Evidence table check
        self.assertIsNotNone(result.evidence_df)
        self.assertEqual(len(result.evidence_df), 2)
        top_share = result.evidence_df.iloc[0]["Share of Total Losses (%)"]
        self.assertAlmostEqual(top_share, (1000.0 / 1500.0) * 100, places=2)

    def test_5_missing_financial_columns_handled_safely(self):
        """5. Missing financial columns are handled safely without crash."""
        df = pd.DataFrame({
            "product_name": ["Item A", "Item B"],
            "category": ["Hardware", "Software"]
        })
        mapping = detect_column_mapping(df)
        result = detect_loss_making_products(df, mapping)

        self.assertFalse(result.is_applicable)
        self.assertEqual(result.total_flagged_count, 0)
        self.assertEqual(len(result.alerts), 0)
        self.assertIn("Insufficient financial data", result.status_message)

    def test_6_empty_dataset_handled_safely(self):
        """6. An empty dataset is handled safely."""
        df = pd.DataFrame()
        mapping = ColumnMapping(product_col="product", revenue_col="revenue", cost_col="cost")
        result = detect_loss_making_products(df, mapping)

        self.assertFalse(result.is_applicable)
        self.assertEqual(result.total_flagged_count, 0)
        self.assertIn("empty", result.status_message.lower())

    def test_7_missing_and_non_numeric_values_safe(self):
        """7. Missing and non-numeric values do not produce misleading results or crash."""
        df = pd.DataFrame({
            "product": ["Widget Alpha", "Widget Alpha", None, "Widget Beta"],
            "revenue": ["$1,000", np.nan, "500", "invalid_str"],
            "cost": [1500.0, 100.0, "300", 200.0]
        })
        mapping = ColumnMapping(product_col="product", revenue_col="revenue", cost_col="cost")
        result = detect_loss_making_products(df, mapping)

        self.assertTrue(result.is_applicable)
        self.assertTrue(len(result.quality_warnings) > 0)
        # Should not crash and should properly evaluate valid numeric values
        flagged_names = [a.product_name for a in result.alerts]
        self.assertIn("Widget Alpha", flagged_names)

    def test_8_zero_revenue_division_by_zero_safety(self):
        """8. Zero revenue does not cause a division-by-zero error in profit-margin calculations."""
        df = pd.DataFrame({
            "product_name": ["Free Tier Deficit"],
            "revenue": [0.0],
            "cost": [250.0]  # Profit: -250.0
        })
        mapping = detect_column_mapping(df)
        result = detect_loss_making_products(df, mapping)

        self.assertTrue(result.is_applicable)
        self.assertEqual(result.total_flagged_count, 1)
        alert = result.alerts[0]
        self.assertEqual(alert.loss_amount, 250.0)
        self.assertIsNone(alert.profit_margin_pct)  # Undefined, NOT ZeroDivisionError
        self.assertIn("zero revenue", alert.explanation.lower())

    def test_9_existing_filters_and_column_mappings_respected(self):
        """9. Existing filters and column mappings are respected."""
        df = pd.DataFrame({
            "sku_code": ["SKU_1", "SKU_2", "SKU_3", "SKU_4"],
            "dept": ["Audio", "Audio", "Cables", "Cables"],
            "gross_turnover": [1000.0, 500.0, 200.0, 800.0],
            "cogs_expense": [1400.0, 400.0, 350.0, 600.0]
            # Profits: SKU_1 = -400 (Audio), SKU_2 = +100 (Audio), SKU_3 = -150 (Cables), SKU_4 = +200 (Cables)
        })
        mapping = ColumnMapping(
            product_col="sku_code",
            category_col="dept",
            revenue_col="gross_turnover",
            cost_col="cogs_expense"
        )

        # Unfiltered: both SKU_1 and SKU_3 flagged
        all_res = detect_loss_making_products(df, mapping)
        self.assertEqual(all_res.total_flagged_count, 2)

        # Filtered to Cables only: only SKU_3 flagged
        cables_df = df[df["dept"] == "Cables"].copy()
        cables_res = detect_loss_making_products(cables_df, mapping)
        self.assertEqual(cables_res.total_flagged_count, 1)
        self.assertEqual(cables_res.alerts[0].product_name, "SKU_3")
        self.assertEqual(cables_res.alerts[0].loss_amount, 150.0)

        # Filtered to Audio only: only SKU_1 flagged
        audio_df = df[df["dept"] == "Audio"].copy()
        audio_res = detect_loss_making_products(audio_df, mapping)
        self.assertEqual(audio_res.total_flagged_count, 1)
        self.assertEqual(audio_res.alerts[0].product_name, "SKU_1")
        self.assertEqual(audio_res.alerts[0].loss_amount, 400.0)

    def test_10_existing_analytics_functions_unbroken(self):
        """10. The new feature does not break existing analytics functions."""
        df = pd.DataFrame({
            "product_name": ["Prod 1", "Prod 2"],
            "revenue": [2000.0, 1000.0],
            "cost": [1200.0, 1500.0]
        })
        mapping = detect_column_mapping(df)

        # Check existing KPI calculation
        kpis = calculate_business_kpis(df, mapping)
        self.assertEqual(kpis.total_revenue, 3000.0)
        self.assertEqual(kpis.total_cost, 2700.0)
        self.assertEqual(kpis.net_profit, 300.0)

        # Check existing product performance
        prod_df = analyze_product_performance(df, mapping, kpis)
        self.assertIsNotNone(prod_df)
        self.assertEqual(len(prod_df), 2)
        loss_makers_status = prod_df[prod_df["Status"].str.contains("Loss-Maker")]
        self.assertEqual(len(loss_makers_status), 1)

        # Check existing recommendations
        recs = generate_business_recommendations(kpis, prod_df)
        self.assertTrue(len(recs) > 0)

    def test_11_unit_cost_compatibility(self):
        """11. Unit cost without quantity is rejected; with quantity, total cost is derived."""
        # Case A: unit_cost without quantity -> rejected with clear message
        df_no_qty = pd.DataFrame({
            "product": ["Widget A"],
            "revenue": [1000.0],
            "unit_cost": [50.0]
        })
        mapping_no_qty = ColumnMapping(product_col="product", revenue_col="revenue", cost_col="unit_cost")
        res_no_qty = detect_loss_making_products(df_no_qty, mapping_no_qty)
        self.assertFalse(res_no_qty.is_applicable)
        self.assertIn("represent unit cost", res_no_qty.status_message)

        # Case B: unit_cost with quantity -> derives total cost safely
        df_with_qty = pd.DataFrame({
            "product": ["Widget A"],
            "revenue": [1000.0],
            "unit_cost": [30.0],
            "quantity": [40]  # Total cost = 30 * 40 = 1200.0 -> loss of 200.0
        })
        mapping_with_qty = ColumnMapping(product_col="product", revenue_col="revenue", cost_col="unit_cost", quantity_col="quantity")
        res_with_qty = detect_loss_making_products(df_with_qty, mapping_with_qty)
        self.assertTrue(res_with_qty.is_applicable)
        self.assertEqual(res_with_qty.total_flagged_count, 1)
        self.assertEqual(res_with_qty.alerts[0].cost, 1200.0)
        self.assertEqual(res_with_qty.alerts[0].loss_amount, 200.0)

    def test_12_offline_ai_briefing_fallback(self):
        """12. Offline AI investigation fallback provides deterministic guidance without Groq API."""
        alert = BusinessAlert(
            product_name="Sample Deficit SKU",
            revenue=5000.0,
            cost=6500.0,
            net_profit=-1500.0,
            loss_amount=1500.0,
            profit_margin_pct=-30.0,
            transactions=10,
            units_sold=50,
            category="Hardware",
            title="Loss-Making Product Detected: Sample Deficit SKU",
            explanation="Cost exceeds revenue by 1,500.00.",
            suggested_action="Review vendor terms and list prices.",
            evidence_summary="Rev: 5000 | Cost: 6500"
        )
        briefing = generate_alert_investigation_briefing([alert], api_key=None, currency_symbol="$")
        self.assertIsNotNone(briefing)
        self.assertIn("Sample Deficit SKU", briefing)
        self.assertIn("$1,500.00", briefing)
        self.assertIn("Supplier Cost Audit", briefing)
        self.assertIn("Promotional Discount", briefing)


if __name__ == "__main__":
    unittest.main()
