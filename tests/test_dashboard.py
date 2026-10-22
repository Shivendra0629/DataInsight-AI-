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
    generate_business_recommendations
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


if __name__ == "__main__":
    unittest.main()
