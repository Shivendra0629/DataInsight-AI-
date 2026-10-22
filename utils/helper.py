"""
Utility helper functions for DataInsight AI.
Provides resilient formatting for currency, percentages, numbers, and dates.
"""

from typing import Any, Optional, Union
import pandas as pd
import numpy as np


def format_currency(
    val: Optional[Union[int, float]],
    symbol: str = "$",
    decimals: int = 2,
    abbreviate: bool = False
) -> str:
    """
    Format a numeric value as a currency string.
    Gracefully handles None, NaN, inf, and zeroes.
    Optionally abbreviates large numbers (e.g. $1.2M, $45.6K).
    """
    if val is None or pd.isna(val) or np.isinf(val):
        return "N/A"
    
    try:
        val = float(val)
    except (ValueError, TypeError):
        return "N/A"
        
    is_negative = val < 0
    abs_val = abs(val)

    if abbreviate:
        if abs_val >= 1_000_000_000:
            formatted = f"{abs_val / 1_000_000_000:.2f}B"
        elif abs_val >= 1_000_000:
            formatted = f"{abs_val / 1_000_000:.2f}M"
        elif abs_val >= 1_000:
            formatted = f"{abs_val / 1_000:.1f}K"
        else:
            formatted = f"{abs_val:.{decimals}f}"
    else:
        formatted = f"{abs_val:,.{decimals}f}"

    prefix = "-" if is_negative else ""
    return f"{prefix}{symbol}{formatted}"


def format_percent(
    val: Optional[Union[int, float]],
    decimals: int = 1,
    include_sign: bool = False
) -> str:
    """
    Format a float as a percentage string (assumes val is already on 0-100 scale).
    """
    if val is None or pd.isna(val) or np.isinf(val):
        return "N/A"
    
    try:
        val = float(val)
    except (ValueError, TypeError):
        return "N/A"

    sign = "+" if (include_sign and val > 0) else ""
    return f"{sign}{val:.{decimals}f}%"


def format_number(
    val: Optional[Union[int, float]],
    decimals: int = 0,
    abbreviate: bool = False
) -> str:
    """
    Format a general numeric value with thousand separators or abbreviations.
    """
    if val is None or pd.isna(val) or np.isinf(val):
        return "N/A"

    try:
        val = float(val)
    except (ValueError, TypeError):
        return "N/A"

    is_negative = val < 0
    abs_val = abs(val)

    if abbreviate:
        if abs_val >= 1_000_000_000:
            formatted = f"{abs_val / 1_000_000_000:.2f}B"
        elif abs_val >= 1_000_000:
            formatted = f"{abs_val / 1_000_000:.2f}M"
        elif abs_val >= 1_000:
            formatted = f"{abs_val / 1_000:.1f}K"
        else:
            formatted = f"{abs_val:.{decimals}f}"
    else:
        if decimals == 0:
            formatted = f"{int(round(abs_val)):,}"
        else:
            formatted = f"{abs_val:,.{decimals}f}"

    prefix = "-" if is_negative else ""
    return f"{prefix}{formatted}"


def clean_column_name(col: Any) -> str:
    """Return a stripped, sanitized column name string."""
    return str(col).strip()
