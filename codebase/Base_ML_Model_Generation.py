import sys
import numpy as np
import matplotlib.pyplot as plt
import Bid_Item_ML_Pipeline as BIML
from concurrent.futures import ProcessPoolExecutor

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Base path for databases
DB_ROOT = Path(r"D:/Estimation Code/Database")
OUTPUT_DIR = Path(r"D:\Estimation Code\BaseML_Models")
TF_DB = DB_ROOT / "time_adjustment_factors.db"
BID_DB = DB_ROOT / "bid_database.db"


def process_bid(bid: str) -> None:
    model = BIML.BidItemModel(bid, str(BID_DB), str(TF_DB))
    model.train()

    feats = model.extractor.extract(bid)
    Xr = np.array(feats['quantity'], float)
    yr = np.array(feats['unit_price'], float)
    mask = (Xr > 0) & (yr > 0)
    Xr, yr = Xr[mask], yr[mask]

    Xlog = np.log(Xr).reshape(-1, 1)
    ylog = np.log(yr).reshape(-1, 1)
    Xtr, Xte, ytr, yte = train_test_split(Xlog, ylog, test_size=0.2, random_state=42)

    quantities = np.exp(Xte.flatten())
    preds = np.array(model.predict(quantities))
    truth = np.exp(yte.flatten())

    inside = np.mean((truth >= preds[:, 0]) & (truth <= preds[:, 2]))
    print(f"{bid}: Coverage of 90% interval on test: {inside * 100:.1f}%")

    model.save_model(directory=OUTPUT_DIR)


def main():
    # Check DB existence
    missing = []
    if not TF_DB.exists():
        missing.append(f"Time DB missing: {TF_DB}")
    if not BID_DB.exists():
        missing.append(f"Bid DB missing: {BID_DB}")
    if missing:
        print("[ERROR] Required DB files missing:\n", "\n".join(missing))
        sys.exit(1)

    # Fetch valid items
    conn = sqlite3.connect(str(BID_DB))
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT Item_Number FROM bid_items")
    all_items = set(row[0] for row in cursor.fetchall())
    cursor.execute("SELECT DISTINCT Item_Number FROM bids")
    items_with_bids = set(row[0] for row in cursor.fetchall())
    cursor.execute("SELECT DISTINCT Item_Number FROM bids WHERE UOM = 'LS'")
    ls_items = set(row[0] for row in cursor.fetchall())
    conn.close()

    valid_items = sorted(all_items & items_with_bids - ls_items)
    print(valid_items)
    print(len(valid_items))

    # Parallel training
    with ProcessPoolExecutor() as executor:
        executor.map(process_bid, valid_items)


if __name__ == "__main__":
    freeze_support()
    main()
