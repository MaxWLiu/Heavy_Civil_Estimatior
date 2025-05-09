#!/usr/bin/env python3
"""
bid_feature_extractor.py

A streamlined script to extract and normalize bid data features
for machine learning models.

Outputs seven NumPy arrays:
- percent_array: bid item total as fraction of the company's (rank’s) total for that contract (across all items)
- unit_price_array: inflation-normalized unit prices
- quantity_array: bid quantities
- rank_array: bid ranks
- contract_total_array: inflation-normalized total bid amount per company per contract
- uom_array: unit of measure for each bid line-item (as numpy array of strings)
- date_array: bid opening dates (as numpy datetime64 array)

Usage:
    - As a script: python bid_feature_extractor.py
    - As a module: import extract_bid_features and call it in your code

Dependencies:
- sqlite3 (standard library)
- datetime (standard library)
- numpy
- Price_Adjuster.get_ccci and NORMALIZATION_DATE_DEFAULT
"""

import sqlite3
from datetime import datetime
import numpy as np
from Price_Adjuster import get_ccci, NORMALIZATION_DATE_DEFAULT

# Configuration
DB_PATH = r"D:\Estimation Code\Database\bid_database.db"
TABLE_NAME = "bids"

# Use provided normalization date
NORMALIZATION_DATE = NORMALIZATION_DATE_DEFAULT
TARGET_CCCI = get_ccci(NORMALIZATION_DATE.month, NORMALIZATION_DATE.year)

def extract_bid_features(item_number: str, rank_min: int = 1, rank_max: int = 10):
    """
    Fetch bids for a given item_number and rank range,
    compute feature arrays for ML models.

    Returns:
        percent_array (np.ndarray): bid share of contract+rank total
        unit_price_array (np.ndarray): inflation-normalized unit prices
        quantity_array (np.ndarray): bid quantities
        rank_array (np.ndarray): bid ranks
        contract_total_array (np.ndarray): inflation-normalized total bid amount per company per contract
        uom_array (np.ndarray of str): unit of measure for each bid line-item
        date_array (np.ndarray of datetime64): bid opening dates
    """
    # 1. Fetch rows for this item and rank range
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT Contract_Number, Bid_Rank, Total,
                   Unit_Price, Qty, UOM, Bid_Open_Date
            FROM {TABLE_NAME}
            WHERE Item_Number = ?
              AND Bid_Rank BETWEEN ? AND ?
        """, (item_number, rank_min, rank_max))
        rows = cursor.fetchall()

    if not rows:
        empty = np.array([], dtype=float)
        empty_uom = np.array([], dtype='<U')
        empty_dates = np.array([], dtype='datetime64')
        return empty, empty, empty, empty, empty, empty_uom, empty_dates

    # 2. Unpack fetched rows into arrays
    contracts, ranks, totals, unit_prices, quantities, uoms, dates = zip(*rows)
    date_array = np.array([datetime.fromisoformat(d) for d in dates], dtype='datetime64[D]')
    ranks = np.array(ranks, dtype=float)
    totals = np.array(totals, dtype=float)
    unit_prices = np.array(unit_prices, dtype=float)
    quantities = np.array(quantities, dtype=float)
    uom_array = np.array(uoms, dtype='<U')  # numpy array of strings

    # 3. Compute aggregate sums per company per contract via CTE
    contract_rank_sums = {}
    cte_query = f"""
        WITH needed AS (
            SELECT DISTINCT Contract_Number, Bid_Rank
            FROM {TABLE_NAME}
            WHERE Item_Number = ?
              AND Bid_Rank BETWEEN ? AND ?
        )
        SELECT b.Contract_Number, b.Bid_Rank, SUM(b.Total) AS total_sum
        FROM {TABLE_NAME} AS b
        JOIN needed AS k
          ON b.Contract_Number = k.Contract_Number
         AND b.Bid_Rank = k.Bid_Rank
        GROUP BY b.Contract_Number, b.Bid_Rank
    """
    with sqlite3.connect(DB_PATH) as conn2:
        cur2 = conn2.cursor()
        cur2.execute(cte_query, (item_number, rank_min, rank_max))
        for cnum, brank, tsum in cur2.fetchall():
            contract_rank_sums[(cnum, int(brank))] = tsum

    # 4. Compute percent share array without warnings
    denom_array = np.array([
        contract_rank_sums.get((contract, int(rank)), np.nan)
        for contract, rank in zip(contracts, ranks)
    ], dtype=float)
    valid = np.isfinite(denom_array) & (denom_array != 0)
    percent_array = np.full_like(totals, np.nan, dtype=float)
    percent_array[valid] = totals[valid] / denom_array[valid]

    # 5. Normalize unit prices by inflation
    norm_factors = []
    normalized_prices = []
    for price, date_str in zip(unit_prices, dates):
        bid_date = datetime.fromisoformat(date_str)
        bid_ccci = get_ccci(bid_date.month, bid_date.year)
        factor = TARGET_CCCI / bid_ccci if bid_ccci > 0 else 1.0
        norm_factors.append(factor)
        normalized_prices.append(price * factor)
    unit_price_array = np.array(normalized_prices, dtype=float)
    norm_factors = np.array(norm_factors, dtype=float)

    # 6. Build contract_total_array and normalize it by the same factors
    raw_contract_totals = np.array([
        contract_rank_sums.get((contract, int(rank)), np.nan)
        for contract, rank in zip(contracts, ranks)
    ], dtype=float)
    contract_total_array = raw_contract_totals * norm_factors

    # 7. Build arrays for quantity and rank
    quantity_array = quantities
    rank_array = ranks

    # 8. Return all feature arrays + date_array
    return (
        percent_array,
        unit_price_array,
        quantity_array,
        rank_array,
        contract_total_array,
        uom_array,
        date_array
    )

if __name__ == "__main__":
    # Example invocation
    item = "100100"
    low_rank, high_rank = 1, 4
    pct, unit, qty, rk, ct, uom, dates = extract_bid_features(item, low_rank, high_rank)
    print("Percent array:", pct)
    print("Unit price array:", unit)
    print("Quantity array:", qty)
    print("Rank array:", rk)
    print("Contract total array:", ct)
    print("UOM array:", uom)
    print("Date array:", dates)
