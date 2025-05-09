#!/usr/bin/env python3
"""
Cost Adjustment and Normalization Module

Optimized for integration with feature pipelines and batch regression inference.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union
from functools import lru_cache
import numpy as np
import warnings
import sys

# Ensure local imports work even if run as script
from pathlib import Path as SysPath
codebase_dir = SysPath("d:/Estimation Code/Codebase").resolve()
if str(codebase_dir) not in sys.path:
    sys.path.insert(0, str(codebase_dir))

# ——— Module Constants ———
DB_PATH_DEFAULT: Path = Path(r"D:\Estimation Code\Database\time_adjustment_factors.db")
NORMALIZATION_DATE_DEFAULT: datetime = datetime(2024, 12, 1)

# Configure warnings: only show each UserWarning once
warnings.simplefilter('once', UserWarning)

# ——— Utility Functions ———

@lru_cache(maxsize=48)
def get_ccci(
    month: int,
    year: int,
    db_path: Union[str, Path] = DB_PATH_DEFAULT
) -> float:
    """
    Fetch the CCCI value for the given month/year from SQLite using 'California_CCCI' table.
    Falls back to Jan 1996 if too early. If missing, retrieves most recent available.
    """
    target_date = f"{year:04d}-{month:02d}-01"
    db_path = Path(db_path)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT VALUE FROM California_CCCI WHERE DATE = ?",
            (target_date,)
        )
        row = cursor.fetchone()

        if row is None:
            if year < 1996 or (year == 1996 and month == 1):
                warnings.warn(
                    "Date older than January 1996: defaulting to Jan 1996 CCCI",
                    UserWarning,
                    stacklevel=2
                )
                cursor = conn.execute(
                    "SELECT VALUE FROM California_CCCI WHERE DATE = '1996-01-01'"
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError("Critical: Jan 1996 fallback CCCI data is missing.")
            else:
                warnings.warn(
                    f"No CCCI data found for {target_date}, using most recent available.",
                    UserWarning,
                    stacklevel=2
                )
                cursor = conn.execute(
                    "SELECT VALUE FROM California_CCCI ORDER BY DATE DESC LIMIT 1"
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError("No CCCI records available in database.")
        return float(row[0])

def adjust_cost_for_inflation(
    original_cost: float,
    ccci_old: float,
    ccci_new: float
) -> float:
    """
    Escalate a single cost from `ccci_old` to `ccci_new`.
    """
    if ccci_old <= 0:
        raise ValueError(f"Invalid historical CCCI: {ccci_old}")
    return original_cost * (ccci_new / ccci_old)

# ——— Normalization API ———

def normalize_prices(
    contract_date: Union[datetime, np.datetime64],
    unit_costs: List[float],
    total_costs: List[float],
    db_path: Union[str, Path] = DB_PATH_DEFAULT,
    normalization_date: datetime = NORMALIZATION_DATE_DEFAULT
) -> Tuple[List[float], List[float]]:
    """
    Normalize costs (unit & total) from contract_date to normalization_date.
    """
    if isinstance(contract_date, np.datetime64):
        contract_date = datetime.utcfromtimestamp(
            contract_date.astype('O') / 1e9
        )

    ccci_old = get_ccci(contract_date.month, contract_date.year, db_path)
    ccci_new = get_ccci(normalization_date.month, normalization_date.year, db_path)

    normalized_units = [
        adjust_cost_for_inflation(u, ccci_old, ccci_new)
        for u in unit_costs
    ]
    normalized_totals = [
        adjust_cost_for_inflation(t, ccci_old, ccci_new)
        for t in total_costs
    ]
    return normalized_units, normalized_totals

# ——— CLI Entry ———

if __name__ == "__main__":
    try:
        from bid_feature_extractor import extract_bid_features
        item_number = "870510"
        bid_rank_range = (1, 2)
        _, unit_prices, _, _, total_costs, _ = extract_bid_features(item_number, *bid_rank_range)
        # For demo purposes, use today's date as contract_date
        contract_date = datetime.today()
        units, totals = normalize_prices(contract_date, list(unit_prices), list(total_costs))
        print("Normalized Unit Costs:", units)
        print("Normalized Total Costs:", totals)
    except Exception as e:
        print(f"[ERROR] Normalization failed: {e}")
