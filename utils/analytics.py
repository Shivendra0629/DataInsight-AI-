"""
Business Analytics & KPI Calculation Engine for DataInsight AI.
Handles automatic column mapping, data validation, business KPI computations,
product performance analysis, and evidence-based recommendations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import re
import warnings
import pandas as pd
import numpy as np


@dataclass
class ColumnMapping:
    """Stores the identified or user-overridden column mappings."""
    date_col: Optional[str] = None
    product_col: Optional[str] = None
    category_col: Optional[str] = None
    revenue_col: Optional[str] = None
    cost_col: Optional[str] = None
    profit_col: Optional[str] = None
    quantity_col: Optional[str] = None
    price_col: Optional[str] = None
    customer_col: Optional[str] = None
    order_id_col: Optional[str] = None
    profit_mode: str = "auto"  # 'direct', 'calculated', 'none'
    derived_revenue: bool = False  # True if revenue was derived from price * quantity

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "Date": self.date_col,
            "Product": self.product_col,
            "Category": self.category_col,
            "Revenue": self.revenue_col,
            "Cost": self.cost_col,
            "Profit": self.profit_col,
            "Quantity": self.quantity_col,
            "Unit Price": self.price_col,
            "Customer": self.customer_col,
            "Order ID": self.order_id_col,
        }


@dataclass
class BusinessKPIs:
    """Contains calculated business metrics and health indicators."""
    total_revenue: Optional[float] = None
    total_cost: Optional[float] = None
    net_profit: Optional[float] = None
    total_loss_negative_records: Optional[float] = None
    negative_records_count: int = 0
    profit_margin_pct: Optional[float] = None
    transaction_count: int = 0
    unique_orders: Optional[int] = None
    unique_products: Optional[int] = None
    unique_categories: Optional[int] = None
    unique_customers: Optional[int] = None
    average_order_value: Optional[float] = None
    is_aov_order_level: bool = False
    revenue_growth_pct: Optional[float] = None
    profit_growth_pct: Optional[float] = None
    growth_period_label: Optional[str] = None
    is_financial: bool = False
    profit_source_explanation: str = ""
    notes_and_assumptions: List[str] = field(default_factory=list)
    quality_warnings: List[str] = field(default_factory=list)


@dataclass
class Recommendation:
    """Structured evidence-based recommendation."""
    title: str
    category: str  # 'Profitability', 'Pricing', 'Growth', 'Risk', 'Data Quality'
    priority: str  # 'High Priority', 'Medium Priority', 'Opportunity', 'Information'
    observation: str
    evidence: str
    business_impact: str
    next_steps: str


# ==============================================================================
# 1. Automatic Column Detection
# ==============================================================================

# Regex patterns prioritized for business field identification
PATTERNS = {
    "order_id": [
        r"^order[_\-\s]?(id|num|number|no|code)?$", r"^invoice[_\-\s]?(id|num|number|no)?$",
        r"^trans(action)?[_\-\s]?(id|num|no)?$", r"^receipt[_\-\s]?(id|no)?$",
        r"^(order|invoice|trans)[_\-\s]?(no|#|code)$"
    ],
    "revenue": [
        r"^(total[_\-\s]?)?revenue$", r"^(total[_\-\s]?)?sales$", r"^turnover$",
        r"^gross[_\-\s]?sales$", r"^(total[_\-\s]?)?amount$", r"^order[_\-\s]?total$",
        r"^extended[_\-\s]?price$", r"^net[_\-\s]?sales$", r"^subtotal$",
        r".*(sales_turnover|sales_revenue|order.*sales|sales.*amount|total.*sales|total.*revenue).*"
    ],
    "cost": [
        r"^cogs$", r"^cost[_\-\s]?of[_\-\s]?goods[_\-\s]?sold$", r"^cost$",
        r"^total[_\-\s]?cost$", r"^expense[s]?$", r"^total[_\-\s]?expense[s]?$",
        r"^purchase[_\-\s]?cost$", r"^cost[_\-\s]?price$", r"^spending$",
        r".*(cost_price|total_cost|total_expense).*"
    ],
    "profit": [
        r"^profit$", r"^net[_\-\s]?profit$", r"^net[_\-\s]?income$",
        r"^gross[_\-\s]?profit$", r"^operating[_\-\s]?profit$", r"^earnings$",
        r"^gain$", r".*(net_profit|profit_margin_amount|profit_amount).*"
    ],
    "price": [
        r"^unit[_\-\s]?price$", r"^price$", r"^rate$", r"^selling[_\-\s]?price$",
        r"^list[_\-\s]?price$", r"^retail[_\-\s]?price$"
    ],
    "quantity": [
        r"^qty$", r"^quantity$", r"^units$", r"^units[_\-\s]?sold$",
        r"^volume$", r"^pieces$", r"^count$", r"^order[_\-\s]?qty$"
    ],
    "date": [
        r"^order[_\-\s]?date$", r"^invoice[_\-\s]?date$", r"^trans(action)?[_\-\s]?date$",
        r"^date$", r"^datetime$", r"^timestamp$", r"^sale[_\-\s]?date$",
        r"^day$", r"^month$", r"^period$"
    ],
    "product": [
        r"^product[_\-\s]?(name|title|desc|description|id|code|num|number)?$",
        r"^item[_\-\s]?(name|title|desc|id|code|num|number)?$",
        r"^product$", r"^item$", r"^sku$", r"^article$", r"^goods$"
    ],
    "category": [
        r"^category([_\-\s]?name)?$", r"^prod(uct)?[_\-\s]?category$", r"^sub[_\-\s]?category$",
        r"^department$", r"^dept$", r"^segment$", r"^product[_\-\s]?line$",
        r"^group$", r"^classification$", r"^type$"
    ],
    "customer": [
        r"^customer[_\-\s]?(name|id|code)?$", r"^client[_\-\s]?(name|id)?$",
        r"^customer$", r"^client$", r"^buyer$", r"^account[_\-\s]?(name|id)?$"
    ]
}


def _match_column(col_name: str, pattern_list: List[str]) -> bool:
    """Check if column name matches any regex in pattern_list."""
    norm = col_name.strip().lower()
    for pattern in pattern_list:
        if re.search(pattern, norm):
            return True
    return False


def is_potential_date_series(s: pd.Series) -> bool:
    """Check if a pandas Series represents dates."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        return False
    sample = s.dropna().head(20)
    if sample.empty:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        return parsed.notna().mean() >= 0.8
    except Exception:
        return False


def detect_column_mapping(df: pd.DataFrame) -> ColumnMapping:
    """
    Intelligently inspects a DataFrame and returns suggested column mappings.
    Considers column names, data types, and value characteristics.
    """
    mapping = ColumnMapping()
    if df is None or df.empty:
        return mapping

    cols = list(df.columns)
    assigned = set()

    def get_candidate(role: str, require_numeric: bool = False, require_date: bool = False) -> Optional[str]:
        for c in cols:
            if c in assigned:
                continue
            if _match_column(str(c), PATTERNS.get(role, [])):
                if require_numeric and not pd.api.types.is_numeric_dtype(df[c]):
                    numeric_test = pd.to_numeric(df[c], errors="coerce")
                    if numeric_test.notna().mean() < 0.6:
                        continue
                if require_date and not is_potential_date_series(df[c]):
                    continue
                return str(c)
        return None

    # Priority 1: Date column
    mapping.date_col = get_candidate("date", require_date=True)
    if mapping.date_col:
        assigned.add(mapping.date_col)
    else:
        for c in cols:
            if c not in assigned and is_potential_date_series(df[c]):
                mapping.date_col = str(c)
                assigned.add(mapping.date_col)
                break

    # Priority 2: Order / Transaction ID
    mapping.order_id_col = get_candidate("order_id")
    if mapping.order_id_col:
        assigned.add(mapping.order_id_col)

    # Priority 3: Revenue / Sales
    mapping.revenue_col = get_candidate("revenue", require_numeric=True)
    if mapping.revenue_col:
        assigned.add(mapping.revenue_col)

    # Priority 4: Cost / Expenses
    mapping.cost_col = get_candidate("cost", require_numeric=True)
    if mapping.cost_col:
        assigned.add(mapping.cost_col)

    # Priority 5: Profit
    mapping.profit_col = get_candidate("profit", require_numeric=True)
    if mapping.profit_col:
        assigned.add(mapping.profit_col)

    # Priority 6: Quantity
    mapping.quantity_col = get_candidate("quantity", require_numeric=True)
    if mapping.quantity_col:
        assigned.add(mapping.quantity_col)

    # Priority 7: Unit Price
    mapping.price_col = get_candidate("price", require_numeric=True)
    if mapping.price_col:
        assigned.add(mapping.price_col)

    # Priority 8: Product
    mapping.product_col = get_candidate("product")
    if mapping.product_col:
        assigned.add(mapping.product_col)

    # Priority 9: Category
    mapping.category_col = get_candidate("category")
    if mapping.category_col:
        assigned.add(mapping.category_col)

    # Priority 10: Customer
    mapping.customer_col = get_candidate("customer")
    if mapping.customer_col:
        assigned.add(mapping.customer_col)

    # Determine default profit mode
    if mapping.profit_col:
        mapping.profit_mode = "direct"
    elif mapping.revenue_col and mapping.cost_col:
        mapping.profit_mode = "calculated"
    else:
        mapping.profit_mode = "none"

    return mapping


# ==============================================================================
# 2. Dataset Validation & Financial Calculation Layer
# ==============================================================================

def prepare_analysis_dataframe(
    df: pd.DataFrame,
    mapping: ColumnMapping
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepares a safe copy of the DataFrame for numerical calculations.
    Does NOT alter user data permanently; safely coerces necessary columns.
    """
    warnings_list: List[str] = []
    if df is None or df.empty:
        return pd.DataFrame(), ["The dataset is completely empty."]

    calc_df = df.copy()

    # Safely coerce numeric columns
    numeric_roles = [
        ("Revenue", mapping.revenue_col),
        ("Cost", mapping.cost_col),
        ("Profit", mapping.profit_col),
        ("Quantity", mapping.quantity_col),
        ("Unit Price", mapping.price_col),
    ]

    for label, col in numeric_roles:
        if col and col in calc_df.columns:
            if not pd.api.types.is_numeric_dtype(calc_df[col]):
                if calc_df[col].dtype == object or isinstance(calc_df[col].dtype, pd.StringDtype):
                    cleaned_series = (
                        calc_df[col].astype(str)
                        .str.replace(r"[\$,\s€£₹]", "", regex=True)
                        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
                    )
                    calc_df[col] = pd.to_numeric(cleaned_series, errors="coerce")
                else:
                    calc_df[col] = pd.to_numeric(calc_df[col], errors="coerce")
                
                null_pct = calc_df[col].isna().mean()
                if null_pct > 0.05:
                    warnings_list.append(
                        f"Column '{col}' ({label}) contained non-numeric entries; "
                        f"{null_pct * 100:.1f}% were treated as zero or omitted from sums."
                    )

    # Derive revenue if revenue is missing but price and quantity are available
    if not mapping.revenue_col and mapping.price_col and mapping.quantity_col:
        calc_df["_derived_revenue"] = calc_df[mapping.price_col].fillna(0) * calc_df[mapping.quantity_col].fillna(0)
        mapping.revenue_col = "_derived_revenue"
        mapping.derived_revenue = True
        warnings_list.append(
            f"Derived total revenue as ({mapping.price_col} × {mapping.quantity_col}) "
            "because no standalone revenue column was mapped."
        )

    # Safely parse date column
    if mapping.date_col and mapping.date_col in calc_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(calc_df[mapping.date_col]):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    calc_df["_parsed_date"] = pd.to_datetime(calc_df[mapping.date_col], errors="coerce", format="mixed")
                invalid_dates = calc_df["_parsed_date"].isna().sum()
                if invalid_dates > 0:
                    warnings_list.append(
                        f"Date column '{mapping.date_col}' had {invalid_dates} invalid or unparseable dates."
                    )
            except Exception as e:
                warnings_list.append(f"Could not parse dates from '{mapping.date_col}': {e}")
                calc_df["_parsed_date"] = pd.NaT
        else:
            calc_df["_parsed_date"] = calc_df[mapping.date_col]

    return calc_df, warnings_list


def calculate_business_kpis(
    df: pd.DataFrame,
    mapping: ColumnMapping
) -> BusinessKPIs:
    """
    Computes all core business metrics according to strict BI standards.
    Separates financial data from non-financial datasets cleanly.
    """
    kpis = BusinessKPIs()
    if df is None or df.empty:
        kpis.notes_and_assumptions.append("Empty dataset provided; metrics cannot be calculated.")
        return kpis

    calc_df, prep_warnings = prepare_analysis_dataframe(df, mapping)
    kpis.quality_warnings.extend(prep_warnings)
    kpis.transaction_count = len(calc_df)

    # Unique dimension counts
    if mapping.order_id_col and mapping.order_id_col in calc_df.columns:
        kpis.unique_orders = int(calc_df[mapping.order_id_col].nunique())
    if mapping.product_col and mapping.product_col in calc_df.columns:
        kpis.unique_products = int(calc_df[mapping.product_col].nunique())
    if mapping.category_col and mapping.category_col in calc_df.columns:
        kpis.unique_categories = int(calc_df[mapping.category_col].nunique())
    if mapping.customer_col and mapping.customer_col in calc_df.columns:
        kpis.unique_customers = int(calc_df[mapping.customer_col].nunique())

    # Revenue calculation
    rev_col = mapping.revenue_col
    if rev_col and rev_col in calc_df.columns:
        kpis.total_revenue = float(calc_df[rev_col].fillna(0).sum())
        kpis.is_financial = True
        if mapping.derived_revenue:
            kpis.notes_and_assumptions.append("Revenue is estimated as (Unit Price × Quantity).")

    # Cost calculation
    cost_col = mapping.cost_col
    if cost_col and cost_col in calc_df.columns:
        kpis.total_cost = float(calc_df[cost_col].fillna(0).sum())
        kpis.is_financial = True
        kpis.notes_and_assumptions.append(
            f"Total Cost is based on mapped column '{cost_col}'. Note that other unmapped operating expenses may exist."
        )

    # Profit calculation & source explanation
    record_profits: Optional[pd.Series] = None

    if mapping.profit_mode == "direct" and mapping.profit_col and mapping.profit_col in calc_df.columns:
        record_profits = calc_df[mapping.profit_col].fillna(0)
        kpis.net_profit = float(record_profits.sum())
        kpis.profit_source_explanation = f"Direct profit column '{mapping.profit_col}'"
        kpis.is_financial = True
    elif mapping.profit_mode == "calculated" and rev_col and cost_col and rev_col in calc_df.columns and cost_col in calc_df.columns:
        record_profits = calc_df[rev_col].fillna(0) - calc_df[cost_col].fillna(0)
        kpis.net_profit = float(record_profits.sum())
        kpis.profit_source_explanation = f"Calculated as Revenue ('{rev_col}') − Cost ('{cost_col}')"
        kpis.is_financial = True
    elif mapping.profit_col and mapping.profit_col in calc_df.columns:
        record_profits = calc_df[mapping.profit_col].fillna(0)
        kpis.net_profit = float(record_profits.sum())
        kpis.profit_source_explanation = f"Direct profit column '{mapping.profit_col}'"
        kpis.is_financial = True
    elif rev_col and cost_col and rev_col in calc_df.columns and cost_col in calc_df.columns:
        record_profits = calc_df[rev_col].fillna(0) - calc_df[cost_col].fillna(0)
        kpis.net_profit = float(record_profits.sum())
        kpis.profit_source_explanation = f"Calculated as Revenue ('{rev_col}') − Cost ('{cost_col}')"
        kpis.is_financial = True
    else:
        kpis.profit_source_explanation = "Profit cannot be calculated (requires either a profit column or both revenue and cost)."

    # Profit Margin %
    if kpis.net_profit is not None and kpis.total_revenue is not None:
        if kpis.total_revenue > 0:
            kpis.profit_margin_pct = float((kpis.net_profit / kpis.total_revenue) * 100)
        elif kpis.total_revenue == 0:
            kpis.profit_margin_pct = 0.0
            kpis.notes_and_assumptions.append("Total revenue is zero; profit margin is set to 0.0%.")
        else:
            kpis.profit_margin_pct = None
            kpis.notes_and_assumptions.append("Total revenue is negative; percentage margin is mathematically undefined.")

    # Total loss from negative-profit records
    if record_profits is not None:
        neg_records = record_profits[record_profits < 0]
        kpis.negative_records_count = int(len(neg_records))
        kpis.total_loss_negative_records = float(neg_records.abs().sum())

    # Average Order Value (AOV)
    if kpis.total_revenue is not None:
        if kpis.unique_orders and kpis.unique_orders > 0:
            kpis.average_order_value = float(kpis.total_revenue / kpis.unique_orders)
            kpis.is_aov_order_level = True
        elif kpis.transaction_count > 0:
            kpis.average_order_value = float(kpis.total_revenue / kpis.transaction_count)
            kpis.is_aov_order_level = False
            kpis.notes_and_assumptions.append("AOV represents average revenue per record/transaction (no order ID mapped).")

    # Trend calculation (period-over-period)
    if "_parsed_date" in calc_df.columns and calc_df["_parsed_date"].notna().sum() >= 2:
        valid_date_df = calc_df.dropna(subset=["_parsed_date"]).sort_values("_parsed_date")
        if not valid_date_df.empty:
            date_range_days = (valid_date_df["_parsed_date"].max() - valid_date_df["_parsed_date"].min()).days
            if date_range_days > 180:
                period_freq = "ME"
                kpis.growth_period_label = "MoM"
            elif date_range_days > 30:
                period_freq = "W"
                kpis.growth_period_label = "WoW"
            else:
                period_freq = "D"
                kpis.growth_period_label = "DoD"

            try:
                valid_date_df = valid_date_df.set_index("_parsed_date")
                agg_dict = {}
                if rev_col and rev_col in valid_date_df.columns:
                    agg_dict[rev_col] = "sum"
                if record_profits is not None:
                    valid_date_df["_record_profit"] = record_profits.values
                    agg_dict["_record_profit"] = "sum"

                if agg_dict:
                    period_df = valid_date_df.resample(period_freq).agg(agg_dict).dropna()
                    if len(period_df) >= 2:
                        last_period = period_df.iloc[-1]
                        prev_period = period_df.iloc[-2]

                        if rev_col and prev_period[rev_col] > 0:
                            kpis.revenue_growth_pct = float(
                                ((last_period[rev_col] - prev_period[rev_col]) / prev_period[rev_col]) * 100
                            )

                        if "_record_profit" in period_df.columns and abs(prev_period["_record_profit"]) > 0:
                            kpis.profit_growth_pct = float(
                                ((last_period["_record_profit"] - prev_period["_record_profit"]) / abs(prev_period["_record_profit"])) * 100
                            )
            except Exception as e:
                kpis.notes_and_assumptions.append(f"Trend calculation notice: {e}")

    # General dataset fallback note
    if not kpis.is_financial:
        kpis.notes_and_assumptions.append(
            "This dataset lacks explicit financial columns (Revenue/Sales/Cost/Profit). "
            "General data distributions, record counts, and categorizations are provided."
        )

    return kpis


# ==============================================================================
# 3. Product & Category Performance Engine
# ==============================================================================

def analyze_product_performance(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    overall_kpis: Optional[BusinessKPIs] = None
) -> Optional[pd.DataFrame]:
    """
    Computes aggregated performance per product including revenue, profit,
    margins, losses, transaction volume, and automated health classification.
    """
    if df is None or df.empty or not mapping.product_col or mapping.product_col not in df.columns:
        return None

    calc_df, _ = prepare_analysis_dataframe(df, mapping)
    prod_col = mapping.product_col
    rev_col = mapping.revenue_col
    cost_col = mapping.cost_col
    profit_col = mapping.profit_col
    qty_col = mapping.quantity_col
    cat_col = mapping.category_col

    # Compute row-level profit
    if mapping.profit_mode == "direct" and profit_col and profit_col in calc_df.columns:
        calc_df["_calc_profit"] = calc_df[profit_col].fillna(0)
    elif rev_col and cost_col and rev_col in calc_df.columns and cost_col in calc_df.columns:
        calc_df["_calc_profit"] = calc_df[rev_col].fillna(0) - calc_df[cost_col].fillna(0)
    elif profit_col and profit_col in calc_df.columns:
        calc_df["_calc_profit"] = calc_df[profit_col].fillna(0)
    else:
        calc_df["_calc_profit"] = np.nan

    grouped = calc_df.groupby(prod_col, dropna=False)

    summary_data = []
    for prod_name, group in grouped:
        p_name = str(prod_name) if pd.notna(prod_name) else "Unknown / Unnamed"
        trans_count = len(group)
        rev = float(group[rev_col].fillna(0).sum()) if (rev_col and rev_col in group.columns) else 0.0
        cost = float(group[cost_col].fillna(0).sum()) if (cost_col and cost_col in group.columns) else 0.0
        
        has_profit = group["_calc_profit"].notna().any()
        profit = float(group["_calc_profit"].fillna(0).sum()) if has_profit else None

        margin_pct = (profit / rev * 100) if (profit is not None and rev > 0) else None

        if has_profit:
            neg_subset = group[group["_calc_profit"] < 0]
            loss = float(neg_subset["_calc_profit"].abs().sum())
            neg_count = len(neg_subset)
        else:
            loss = 0.0
            neg_count = 0

        units = float(group[qty_col].fillna(0).sum()) if (qty_col and qty_col in group.columns) else None
        cat_val = str(group[cat_col].mode().iloc[0]) if (cat_col and cat_col in group.columns and not group[cat_col].dropna().empty) else "N/A"

        summary_data.append({
            "Product": p_name,
            "Category": cat_val,
            "Revenue": rev,
            "Cost": cost,
            "Net Profit": profit,
            "Profit Margin (%)": margin_pct,
            "Loss from Negatives": loss,
            "Negative Records": neg_count,
            "Units Sold": units,
            "Transactions": trans_count
        })

    result_df = pd.DataFrame(summary_data)
    if result_df.empty:
        return None

    if result_df["Net Profit"].notna().any():
        result_df.sort_values(by="Net Profit", ascending=False, inplace=True)
    else:
        result_df.sort_values(by="Revenue", ascending=False, inplace=True)

    median_rev = result_df["Revenue"].median() if not result_df["Revenue"].empty else 0
    q75_rev = result_df["Revenue"].quantile(0.75) if not result_df["Revenue"].empty else 0
    overall_margin = overall_kpis.profit_margin_pct if (overall_kpis and overall_kpis.profit_margin_pct) else 15.0

    def classify_row(r: pd.Series) -> str:
        p = r["Net Profit"]
        m = r["Profit Margin (%)"]
        rev = r["Revenue"]

        if p is not None and p < 0:
            return "Loss-Maker 🔴"
        if rev >= q75_rev and m is not None and (m < 8.0 or m < (overall_margin * 0.5)):
            return "High Rev / Low Margin ⚠️"
        if p is not None and p > 0 and rev >= q75_rev:
            return "Top Performer 🟢"
        if p is not None and p > 0:
            return "Profitable 🔵"
        if rev > median_rev:
            return "Active Volume ⚪"
        return "Low Volume ⚪"

    result_df["Status"] = result_df.apply(classify_row, axis=1)
    result_df.reset_index(drop=True, inplace=True)
    return result_df


def analyze_category_performance(
    df: pd.DataFrame,
    mapping: ColumnMapping
) -> Optional[pd.DataFrame]:
    """Computes aggregated performance per category."""
    if df is None or df.empty or not mapping.category_col or mapping.category_col not in df.columns:
        return None

    calc_df, _ = prepare_analysis_dataframe(df, mapping)
    cat_col = mapping.category_col
    rev_col = mapping.revenue_col
    cost_col = mapping.cost_col
    profit_col = mapping.profit_col

    if mapping.profit_mode == "direct" and profit_col and profit_col in calc_df.columns:
        calc_df["_calc_profit"] = calc_df[profit_col].fillna(0)
    elif rev_col and cost_col and rev_col in calc_df.columns and cost_col in calc_df.columns:
        calc_df["_calc_profit"] = calc_df[rev_col].fillna(0) - calc_df[cost_col].fillna(0)
    elif profit_col and profit_col in calc_df.columns:
        calc_df["_calc_profit"] = calc_df[profit_col].fillna(0)
    else:
        calc_df["_calc_profit"] = np.nan

    grouped = calc_df.groupby(cat_col, dropna=False)
    summary_data = []

    for cat_name, group in grouped:
        c_label = str(cat_name) if pd.notna(cat_name) else "Uncategorized"
        rev = float(group[rev_col].fillna(0).sum()) if (rev_col and rev_col in group.columns) else 0.0
        cost = float(group[cost_col].fillna(0).sum()) if (cost_col and cost_col in group.columns) else 0.0
        has_profit = group["_calc_profit"].notna().any()
        profit = float(group["_calc_profit"].fillna(0).sum()) if has_profit else None
        margin_pct = (profit / rev * 100) if (profit is not None and rev > 0) else None
        
        prod_count = group[mapping.product_col].nunique() if (mapping.product_col and mapping.product_col in group.columns) else len(group)

        summary_data.append({
            "Category": c_label,
            "Revenue": rev,
            "Cost": cost,
            "Net Profit": profit,
            "Profit Margin (%)": margin_pct,
            "Products": prod_count,
            "Transactions": len(group)
        })

    cat_df = pd.DataFrame(summary_data)
    if cat_df.empty:
        return None

    if cat_df["Net Profit"].notna().any():
        cat_df.sort_values(by="Net Profit", ascending=False, inplace=True)
    else:
        cat_df.sort_values(by="Revenue", ascending=False, inplace=True)

    cat_df.reset_index(drop=True, inplace=True)
    return cat_df


# ==============================================================================
# 4. Transparent, Evidence-Based Recommendations Engine
# ==============================================================================

def generate_business_recommendations(
    kpis: BusinessKPIs,
    product_df: Optional[pd.DataFrame] = None,
    category_df: Optional[pd.DataFrame] = None,
    currency_symbol: str = "$"
) -> List[Recommendation]:
    """
    Evaluates verified analytical findings against transparent decision rules.
    Outputs structured recommendations formatted with:
    1. Observation
    2. Supporting Evidence
    3. Business Impact
    4. Actionable Next Steps
    """
    recs: List[Recommendation] = []

    # Rule 1: Negative Profit Products (Loss Makers)
    if product_df is not None and not product_df.empty and "Net Profit" in product_df.columns:
        loss_makers = product_df[product_df["Net Profit"] < 0]
        if not loss_makers.empty:
            total_prod_loss = abs(loss_makers["Net Profit"].sum())
            worst_3 = loss_makers.sort_values("Net Profit", ascending=True).head(3)
            worst_names = ", ".join([f"'{row['Product']}' ({currency_symbol}{abs(row['Net Profit']):,.2f})" for _, row in worst_3.iterrows()])

            recs.append(Recommendation(
                title=f"Address {len(loss_makers)} Unprofitable Products Eradicating {currency_symbol}{total_prod_loss:,.2f} in Profits",
                category="Profitability",
                priority="High Priority",
                observation=f"{len(loss_makers)} product(s) are yielding negative cumulative net profit.",
                evidence=f"Total aggregate net loss is {currency_symbol}{total_prod_loss:,.2f}. Highest loss contributors: {worst_names}.",
                business_impact="Direct drag on company operating margin and cash flow. Selling these items in their current cost structure erodes overall business gains.",
                next_steps="Investigate whether negative returns are caused by steep promotional discounting, unabsorbed shipping/logistics expenses, or recent supplier COGS increases. Do not immediately delist; verify if they drive core customer acquisition or require price repricing."
            ))

    # Rule 2: High Revenue, Low Margin Products
    if product_df is not None and not product_df.empty and "Profit Margin (%)" in product_df.columns:
        valid_margins = product_df[product_df["Profit Margin (%)"].notna() & (product_df["Revenue"] > 0)]
        if not valid_margins.empty:
            rev_75 = valid_margins["Revenue"].quantile(0.70)
            target_margin = (kpis.profit_margin_pct * 0.6) if (kpis.profit_margin_pct and kpis.profit_margin_pct > 0) else 8.0
            thin_margin_items = valid_margins[
                (valid_margins["Revenue"] >= rev_75) & 
                (valid_margins["Profit Margin (%)"] < target_margin) & 
                (valid_margins["Profit Margin (%)"] >= 0)
            ]

            if not thin_margin_items.empty:
                sample_item = thin_margin_items.sort_values("Revenue", ascending=False).iloc[0]
                recs.append(Recommendation(
                    title=f"Margin Optimization Needed for {len(thin_margin_items)} High-Volume Products",
                    category="Pricing",
                    priority="Medium Priority",
                    observation=f"{len(thin_margin_items)} top-selling product(s) represent high commercial volume but generate below-average profit margins (<{target_margin:.1f}%).",
                    evidence=f"Top example: '{sample_item['Product']}' generates {currency_symbol}{sample_item['Revenue']:,.2f} in sales but only {sample_item['Profit Margin (%)']:.1f}% net margin.",
                    business_impact="High operational and inventory exposure with minimal margin buffer against inflationary costs or shipping fee fluctuations.",
                    next_steps="Audit supplier price breaks for bulk purchases, test selective price elasticity increases of 2-5%, or package these items into higher-margin bundles."
                ))

    # Rule 3: Record-Level Losses (Returns / Discounts)
    if kpis.total_loss_negative_records and kpis.total_loss_negative_records > 0:
        loss_pct_of_rev = (kpis.total_loss_negative_records / kpis.total_revenue * 100) if (kpis.total_revenue and kpis.total_revenue > 0) else 0
        if loss_pct_of_rev > 2.0 or kpis.total_loss_negative_records > 1000:
            recs.append(Recommendation(
                title=f"Mitigate {currency_symbol}{kpis.total_loss_negative_records:,.2f} in Transaction-Level Losses",
                category="Risk",
                priority="High Priority",
                observation=f"{kpis.negative_records_count} transaction record(s) posted negative net margin.",
                evidence=f"Total negative transaction loss sums to {currency_symbol}{kpis.total_loss_negative_records:,.2f} ({loss_pct_of_rev:.1f}% of total gross revenue).",
                business_impact="Indicates systemic post-sale leakages such as product returns, refunds, damaged transit allowances, or sub-cost promotional clearance.",
                next_steps="Segment transaction logs by return reason or discount coupon codes to isolate whether losses are concentrated in specific warehouse locations, carriers, or sales reps."
            ))

    # Rule 4: Revenue Trend Deceleration / Contraction
    if kpis.revenue_growth_pct is not None and kpis.revenue_growth_pct < -5.0:
        recs.append(Recommendation(
            title=f"Revenue Contraction Warning ({kpis.revenue_growth_pct:.1f}% {kpis.growth_period_label or 'Period-over-Period'})",
            category="Growth",
            priority="High Priority",
            observation=f"Latest period revenue dropped by {abs(kpis.revenue_growth_pct):.1f}% relative to the preceding comparable period.",
            evidence=f"Recorded period growth stands at {kpis.revenue_growth_pct:.1f}%.",
            business_impact="Top-line growth trajectory is contracting, which could compress operating leverage.",
            next_steps="Review conversion funnels, seasonal calendar shifts, marketing spend adjustments, or competitor promotions during this timeframe."
        ))
    elif kpis.revenue_growth_pct is not None and kpis.revenue_growth_pct > 15.0:
        recs.append(Recommendation(
            title=f"Capitalize on Strong Momentum ({kpis.revenue_growth_pct:+.1f}% {kpis.growth_period_label or 'Period-over-Period'})",
            category="Growth",
            priority="Opportunity",
            observation=f"Recent sales surged by {kpis.revenue_growth_pct:.1f}% compared to the prior period.",
            evidence=f"Top-line expansion is outpacing historical baseline trends.",
            business_impact="Demonstrates growing market demand and favorable product-market resonance.",
            next_steps="Ensure inventory replenishment lead times are sufficient to avoid stockouts on fast-moving SKUs during this growth cycle."
        ))

    # Rule 5: Product Revenue Concentration (Pareto 80/20 Rule)
    if product_df is not None and len(product_df) >= 5:
        sorted_rev = product_df.sort_values("Revenue", ascending=False)
        total_rev = sorted_rev["Revenue"].sum()
        if total_rev > 0:
            top_20_count = max(1, int(len(sorted_rev) * 0.2))
            top_20_rev = sorted_rev.head(top_20_count)["Revenue"].sum()
            concentration_pct = (top_20_rev / total_rev) * 100

            if concentration_pct >= 75.0:
                recs.append(Recommendation(
                    title=f"High Revenue Concentration Risk (Top {concentration_pct:.0f}% from {top_20_count} Products)",
                    category="Risk",
                    priority="Medium Priority",
                    observation=f"The top {top_20_count} product(s) ({top_20_count / len(sorted_rev) * 100:.0f}% of catalog) contribute {concentration_pct:.1f}% of total business sales.",
                    evidence=f"Top {top_20_count} SKUs account for {currency_symbol}{top_20_rev:,.2f} of {currency_symbol}{total_rev:,.2f}.",
                    business_impact="Heightened business vulnerability if a key supplier faces delays or if competitor alternatives emerge for these specific items.",
                    next_steps="Develop secondary product lines, expand cross-selling initiatives, and negotiate long-term supply safety stocks for the core revenue drivers."
                ))

    # Rule 6: Non-financial / General Dataset fallback
    if not kpis.is_financial:
        recs.append(Recommendation(
            title="Map Financial Fields for Profitability Insights",
            category="Data Quality",
            priority="Information",
            observation="The uploaded dataset does not contain designated Revenue, Cost, or Profit columns.",
            evidence="Currently displaying record counts and general distributions.",
            business_impact="Financial health KPIs, margin calculations, and unit economics cannot be evaluated.",
            next_steps="Ensure your dataset contains at least sales revenue, selling price, or cost columns to unlock executive profitability metrics."
        ))

    return recs
