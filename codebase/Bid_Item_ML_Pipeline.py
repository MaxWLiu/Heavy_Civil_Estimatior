import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

import numpy as np
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

class TimeSeriesFactorDB:
    """Handles retrieval of time-based adjustment factors from a SQLite database."""
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Time factor database not found at {db_path}")
        self._column_mapping: Dict[str, Dict[str, str]] = {}

    def _resolve_table_columns(self, table: str) -> Dict[str, str]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(f"PRAGMA table_info('{table}')")
            cols = cur.fetchall()
        names = [info[1] for info in cols]
        types = {info[1]: info[2].upper() for info in cols}
        date_cols = [n for n in names if 'date' in n.lower()]
        date_col = date_cols[0] if date_cols else names[0]
        numeric_cols = [n for n in names
                        if n != date_col and types.get(n, '') in ('REAL','INTEGER','NUMERIC','FLOAT','DOUBLE')]
        value_col = numeric_cols[0] if numeric_cols else ([n for n in names if n != date_col][0])
        return {'date_col': date_col, 'value_col': value_col}

    def _get_cols(self, table: str) -> Dict[str, str]:
        if table not in self._column_mapping:
            self._column_mapping[table] = self._resolve_table_columns(table)
        return self._column_mapping[table]

    @lru_cache(maxsize=120)
    def get_factor(self, table: str, year: int, month: int, allow_prior: bool = True) -> Optional[float]:
        cols = self._get_cols(table)
        date_col = cols['date_col']; value_col = cols['value_col']
        target_date = f"{year:04d}-{month:02d}-01"
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT {value_col} FROM {table} WHERE {date_col} = ?", (target_date,))
            row = cur.fetchone()
            if row:
                return float(row[0])
            if allow_prior:
                cur.execute(
                    f"SELECT {value_col} FROM {table} WHERE {date_col} <= ? ORDER BY {date_col} DESC LIMIT 1",
                    (target_date,)
                )
                row = cur.fetchone()
                if row:
                    return float(row[0])
        return None

    def get_ccci(self, year: int, month: int) -> float:
        table = "California_CCCI"
        val = self.get_factor(table, year, month)
        if val is not None:
            return val
        # fallback to Jan 1996
        val = self.get_factor(table, 1996, 1, allow_prior=False)
        if val is not None:
            return val
        raise ValueError(f"No CCCI data for {year}-{month:02d} and no fallback")

    def get_other_indices(self, year: int, month: int) -> Dict[str, float]:
        tables = [
            "Avg_Hourly_Earnings_Construction",
            "Construction_Materials_PPI",
            "Hours_Worked_Construction",
            "Total_Public_Construction_Spending",
        ]
        results: Dict[str, float] = {}
        for tbl in tables:
            val = self.get_factor(tbl, year, month)
            if val is not None:
                results[tbl] = val
        return results

class BidFeatureExtractor:
    """Extracts features for bid items from the bids database."""
    def __init__(self, bids_db_path: str, time_factor_db: TimeSeriesFactorDB,
                 normalization_date: datetime = datetime(2024, 12, 1)):
        self.bids_db_path = Path(bids_db_path)
        if not self.bids_db_path.exists():
            raise FileNotFoundError(f"Bids database not found at {bids_db_path}")
        self.time_factor_db = time_factor_db
        y, m = normalization_date.year, normalization_date.month
        self.target_ccci = self.time_factor_db.get_ccci(y, m)
        self.normalization_date = normalization_date

    def extract(self, item_number: str, rank_min: int = 1, rank_max: int = 10
                ) -> Dict[str, np.ndarray]:
        with sqlite3.connect(self.bids_db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT Contract_Number, Bid_Rank, Qty, Unit_Price, Total, UOM, Bid_Open_Date, MOAB, TRO
                FROM bids
                WHERE Item_Number = ? AND Bid_Rank BETWEEN ? AND ?
            """, (item_number, rank_min, rank_max))
            rows = cur.fetchall()
        if not rows:
            return {k: np.array([]) for k in (
                'percent','quantity','unit_price','item_total','contract_total','rank','uom','date','is_tro','is_moab')}
        contracts, ranks, qtys, unit_prices, totals, uoms, dates, moab_vals, tro_vals = zip(*rows)
        contracts = np.array(contracts); ranks = np.array(ranks, int)
        qtys = np.array(qtys, float); unit_prices = np.array(unit_prices, float)
        totals = np.array(totals, float)
        uoms = np.array(uoms, dtype='<U5')
        date_objs = [datetime.fromisoformat(d) for d in dates]
        date_array = np.array([np.datetime64(f"{d.year:04d}-{d.month:02d}", 'M') for d in date_objs])
        # normalize prices to reference date
        norms = []
        norm_unit = []
        norm_item = []
        for price, item_total, d in zip(unit_prices, totals, date_objs):
            y,m = d.year, d.month
            if m == 12: y+=1; m=1
            else: m+=1
            ccci = self.time_factor_db.get_ccci(y,m)
            factor = self.target_ccci / ccci if ccci>0 else 1.0
            norms.append(factor)
            norm_unit.append(price * factor)
            norm_item.append(item_total * factor)
        norm_factors = np.array(norms)
        unit_price_array = np.array(norm_unit)
        item_total_array = np.array(norm_item)
        # contract totals
        sums: Dict[Tuple[str,int], float] = {}
        for c, r, tot in zip(contracts, ranks, totals):
            sums[(c,int(r))] = sums.get((c,int(r)),0.0) + float(tot)
        contract_total_array = np.array([sums[(c,int(r))] for c,r in zip(contracts,ranks)]) * norm_factors
        # percent of contract
        percent_array = np.full_like(item_total_array, np.nan)
        valid = (contract_total_array!=0) & ~np.isnan(contract_total_array)
        percent_array[valid] = item_total_array[valid] / contract_total_array[valid]
        # boolean flags
        tro_flags = np.array(tro_vals, bool)
        moab_flags = np.array(moab_vals, bool)
        # other indices
        other_idxs: List[Dict[str,float]] = []
        for d in date_objs:
            y,m = d.year, d.month
            if m==12: y+=1; m=1
            else: m+=1
            other_idxs.append(self.time_factor_db.get_other_indices(y,m))
        return {
            'percent': percent_array,
            'quantity': qtys,
            'unit_price': unit_price_array,
            'item_total': item_total_array,
            'contract_total': contract_total_array,
            'rank': ranks.astype(float),
            'uom': uoms,
            'date': date_array,
            'is_tro': tro_flags,
            'is_moab': moab_flags,
            'indices': other_idxs
        }

class QuantileNet(nn.Module):
    """Neural network with 3 outputs for 5th, 50th, 95th quantile estimates."""
    def __init__(self, in_dim: int, h1=40, h_o=120, h_l=40, dropout=0.1, dropout_o=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h_o), nn.ReLU(), nn.Dropout(dropout_o),
            nn.Linear(h_o, h_o), nn.ReLU(), nn.Dropout(dropout_o),
            nn.Linear(h_o, h_o), nn.ReLU(), nn.Dropout(dropout_o),
            nn.Linear(h_o, h_l), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h_l, 3)
        )
    def forward(self, x):
        base = self.net(x)                   # shape: [batch,3]
        q5  = base[:,0]
        q50 = q5 + F.softplus(base[:,1])
        q95 = q50 + F.softplus(base[:,2])
        return torch.stack([q5, q50, q95], dim=1)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes sum of pinball (quantile) losses for quantiles [0.05,0.5,0.95].
    pred: [batch,3], target: [batch,1]
    """
    taus = [0.05, 0.5, 0.95]
    losses = []
    for i, tau in enumerate(taus):
        err = target - pred[:, i].view(-1,1)
        losses.append(torch.max((tau-1)*err, tau*err).mean())
    return sum(losses)

class BidItemModel:
    """Manages feature extraction, training, prediction, and saving for bid items."""
    def __init__(self, item_number: str, bids_db_path: str, time_factor_db_path: str,
                 normalization_date: datetime = datetime(2024, 12, 1)):
        self.item_number = item_number
        self.time_db = TimeSeriesFactorDB(time_factor_db_path)
        self.extractor = BidFeatureExtractor(bids_db_path, self.time_db, normalization_date)
        self.model_type: Optional[str] = None
        self.model = None
        self.scaler_X: Optional[StandardScaler] = None
        self.scaler_y: Optional[StandardScaler] = None
        self.ensemble_models: List[LinearRegression] = []
        self.trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, rank_min: int = 1, rank_max: int = 10) -> None:
        feats = self.extractor.extract(self.item_number, rank_min, rank_max)
        X = feats['quantity']; y = feats['unit_price']
        if X.size==0 or y.size==0:
            raise ValueError(f"No data for item {self.item_number}")
        # filter positives
        mask = (X>0)&(y>0)
        Xf = X[mask]; yf = y[mask]
        if Xf.size==0:
            raise ValueError(f"No positive data for item {self.item_number}")
        X_log = np.log(Xf).reshape(-1,1)
        y_log = np.log(yf).reshape(-1,1)
        # scale
        self.scaler_X = StandardScaler(); X_scaled = self.scaler_X.fit_transform(X_log)
        self.scaler_y = StandardScaler(); y_scaled = self.scaler_y.fit_transform(y_log)
        N = X_scaled.shape[0]
        # choose branch: quantile regression vs ensemble fallback
        threshold = 50  # fallback for <=50 samples
        if N > threshold:
            self.model_type = 'quantile_nn'
            model = QuantileNet(X_scaled.shape[1]).double().to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
            # split data
            X_t, X_v, y_t, y_v = train_test_split(
                X_scaled, y_scaled, test_size=0.2, random_state=42
            )
            # dataloaders
            train_ds = torch.utils.data.TensorDataset(
                torch.from_numpy(X_t).double(), torch.from_numpy(y_t).double()
            )
            val_ds = torch.utils.data.TensorDataset(
                torch.from_numpy(X_v).double(), torch.from_numpy(y_v).double()
            )
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
            val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
            # track losses
            self.train_losses = {'q5':[], 'q50':[], 'q95':[]}
            self.val_losses   = {'q5':[], 'q50':[], 'q95':[]}
            epochs = 100
            for epoch in range(1, epochs+1):
                # training epoch
                model.train()
                running = {'q5':0.0,'q50':0.0,'q95':0.0, 'count':0}
                for xb, yb in train_loader:
                    xb=xb.to(self.device); yb=yb.to(self.device)
                    optimizer.zero_grad()
                    preds = model(xb)                # [batch,3]
                    loss = pinball_loss(preds, yb)
                    # accumulate individual losses for logging
                    for i, key in enumerate(['q5','q50','q95']):
                        err = yb - preds[:,i].view(-1,1)
                        lq = torch.max((( [0.05,0.5,0.95][i] )-1)*err,
                                       ([0.05,0.5,0.95][i])*err).sum().item()
                        running[key] += lq
                    running['count'] += xb.size(0)
                    loss.backward(); optimizer.step()
                # compute avg train losses
                for key in ['q5','q50','q95']:
                    self.train_losses[key].append(running[key]/running['count'])
                # validation
                model.eval()
                val_running = {'q5':0.0,'q50':0.0,'q95':0.0,'count':0}
                with torch.no_grad():
                    for xb_v, yb_v in val_loader:
                        xb_v=xb_v.to(self.device); yb_v=yb_v.to(self.device)
                        vp = model(xb_v)
                        for i, key in enumerate(['q5','q50','q95']):
                            err = yb_v - vp[:,i].view(-1,1)
                            lq = torch.max((( [0.05,0.5,0.95][i] )-1)*err,
                                           ([0.05,0.5,0.95][i])*err).sum().item()
                            val_running[key] += lq
                        val_running['count'] += xb_v.size(0)
                for key in ['q5','q50','q95']:
                    self.val_losses[key].append(val_running[key]/val_running['count'])
                if epoch % 20 == 0 or epoch==epochs:
                    print(f"Epoch {epoch}: Train loss q5={self.train_losses['q5'][-1]:.4f}, q50={self.train_losses['q50'][-1]:.4f}, q95={self.train_losses['q95'][-1]:.4f} | "
                          f"Val loss q5={self.val_losses['q5'][-1]:.4f}, q50={self.val_losses['q50'][-1]:.4f}, q95={self.val_losses['q95'][-1]:.4f}")
            self.model = model
        else:
            # ensemble fallback with bootstrap linear models
            self.model_type = 'ensemble'
            rng = np.random.RandomState(42)
            B = 100
            self.ensemble_models = []
            for _ in range(B):
                idxs = rng.choice(N, size=N, replace=True)
                lr = LinearRegression()
                lr.fit(X_scaled[idxs], y_scaled[idxs].ravel())
                self.ensemble_models.append(lr)
            print(f"[INFO] Ensemble fallback: trained {B} linear models on bootstrap samples of {N} points.")
        self.trained = True
        print(f"[INFO] Completed training for item {self.item_number}. Model type: {self.model_type}")

    def predict(self, quantity: float):
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")
        q_arr = np.array(quantity, float, ndmin=1)
        out = np.empty((q_arr.size, 3), float)
        out[:] = np.nan
        valid = (q_arr>0) & np.isfinite(q_arr)
        if np.any(valid):
            X_log = np.log(q_arr[valid]).reshape(-1,1)
            X_scaled = self.scaler_X.transform(X_log)
            if self.model_type == 'quantile_nn':
                self.model.eval()
                with torch.no_grad():
                    t = torch.from_numpy(X_scaled).double().to(self.device)
                    preds = self.model(t).cpu().numpy()   # scaled log-price
                # invert scaling and log
                unscaled = []
                for i in range(3):
                    col = preds[:,i].reshape(-1,1)
                    col_inv = self.scaler_y.inverse_transform(col)
                    unscaled.append(col_inv)
                log_preds = np.hstack(unscaled)     # [n,3]
                price_preds = np.exp(log_preds)
                out[valid] = price_preds
            else:  # ensemble fallback
                preds = []
                for lr in self.ensemble_models:
                    p = lr.predict(X_scaled)         # scaled log-price
                    preds.append(p)
                preds = np.stack(preds, axis=1)    # [n,B]
                q5 = np.percentile(preds, 5, axis=1)
                q50= np.percentile(preds, 50,axis=1)
                q95= np.percentile(preds, 95,axis=1)
                # invert scaling manually
                mean = self.scaler_y.mean_[0]
                scale= self.scaler_y.scale_[0]
                log_q5  = q5*scale + mean
                log_q50 = q50*scale + mean
                log_q95 = q95*scale + mean
                price_preds = np.vstack([np.exp(log_q5), np.exp(log_q50), np.exp(log_q95)]).T
                out[valid] = price_preds
        # return single tuple if one input
        return tuple(out[0]) if out.shape[0]==1 else out.tolist()

    def save_model(self, directory: str = ".") -> Path:
        if not self.trained:
            raise RuntimeError("No trained model to save.")
        date_str = datetime.now().strftime("%Y%m%d")
        dpath = Path(directory); dpath.mkdir(parents=True, exist_ok=True)
        if self.model_type == 'quantile_nn':
            file_path = dpath / f"{self.item_number}_quantile.pt"
            torch.save({
                'model_state': self.model.state_dict(),
                'scaler_X_mean': self.scaler_X.mean_,
                'scaler_X_scale': self.scaler_X.scale_,
                'scaler_y_mean': self.scaler_y.mean_,
                'scaler_y_scale': self.scaler_y.scale_,
                'input_dim': self.scaler_X.mean_.shape[0]
            }, file_path)
        else:
            # save ensemble
            file_path = dpath / f"{self.item_number}_ensemble.pt"
            joblib.dump({
                'ensemble_models': self.ensemble_models,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y
            }, file_path)
        print(f"[INFO] Saved model to {file_path}")
        return file_path
    
    @classmethod

    def load_from_pt(cls,
                     item_number: str,
                     pt_path: str,
                     bids_db_path: str,
                     time_factor_db_path: str,
                     normalization_date: datetime = datetime(2024, 12, 1)) -> "BidItemModel":
        """
        Load a quantile_nn model from a .pt checkpoint and reconstruct scalers.
        """
        # Instantiate empty model object
        self = cls(item_number, bids_db_path, time_factor_db_path, normalization_date)
        # Load checkpoint
        state = torch.load(pt_path, map_location=self.device)

        # Reconstruct scalers
        self.scaler_X = StandardScaler()
        self.scaler_X.mean_ = state['scaler_X_mean']
        self.scaler_X.scale_ = state['scaler_X_scale']
        self.scaler_y = StandardScaler()
        self.scaler_y.mean_ = state['scaler_y_mean']
        self.scaler_y.scale_ = state['scaler_y_scale']

        # Rebuild and load model
        self.model_type = 'quantile_nn'
        self.model = QuantileNet(in_dim=state['input_dim']).double().to(self.device)
        self.model.load_state_dict(state['model_state'])
        self.model.eval()

        self.trained = True
        return self
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    # Configuration
    item = "153121"
    tf_db = DB_ROOT / "time_adjustment_factors.db"
    bid_db = DB_ROOT / "bid_database.db"
    pt_file = Path(f"./{item}_quantile.pt")

    # Validate DB exists
    errors = []
    if not tf_db.exists():
        errors.append(f"Time DB missing: {tf_db}")
    if not bid_db.exists():
        errors.append(f"Bid DB missing: {bid_db}")
    if errors:
        print("[ERROR]\n" + "\n".join(errors))
        sys.exit(1)

    # If checkpoint exists, load; otherwise train & save
    if pt_file.exists():
        model = BidItemModel.load_from_pt(
            item_number=item,
            pt_path=str(pt_file),
            bids_db_path=str(bid_db),
            time_factor_db_path=str(tf_db)
        )
    else:
        model = BidItemModel(item, str(bid_db), str(tf_db))
        model.train()
        model.save_model(directory=".")

    # Extract and filter data
    feats = model.extractor.extract(item)
    Xr = np.array(feats['quantity'], float)
    yr = np.array(feats['unit_price'], float)
    mask = (Xr > 0) & (yr > 0)
    Xr, yr = Xr[mask], yr[mask]

    # Held-out split for coverage
    Xlog = np.log(Xr).reshape(-1, 1)
    ylog = np.log(yr).reshape(-1, 1)
    Xtr, Xte, ytr, yte = train_test_split(Xlog, ylog, test_size=0.2, random_state=42)
    quantities = np.exp(Xte.flatten())
    preds_test = np.array(model.predict(quantities))
    truth = np.exp(yte.flatten())
    coverage = np.mean((truth >= preds_test[:, 0]) & (truth <= preds_test[:, 2]))
    print(f"Coverage of 90% interval on test: {coverage*100:.1f}%")

    # Prepare grid predictions
    grid = np.linspace(Xr.min(), Xr.max(), 200)
    p5, p50, p95 = np.array(model.predict(grid)).T

    # Plot unlogged vs. logged in subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Unlogged plot
    axes[0].scatter(Xr, yr, alpha=0.3, label='Data')
    axes[0].plot(grid, p50, label='Median')
    axes[0].fill_between(grid, p5, p95, alpha=0.2, label='5-95% interval')
    axes[0].set_xlabel('Quantity')
    axes[0].set_ylabel('Unit Price')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title('Unlogged Quantity vs Unit Price')

    # Logged plot
    axes[1].scatter(np.log(Xr), np.log(yr), alpha=0.3, label='Data')
    axes[1].plot(np.log(grid), np.log(p50), label='Median')
    axes[1].fill_between(np.log(grid), np.log(p5), np.log(p95), alpha=0.2, label='5-95% interval')
    axes[1].set_xlabel('log(Quantity)')
    axes[1].set_ylabel('log(Unit Price)')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_title('Log-Transformed Quantity vs Unit Price')

    plt.tight_layout()
    plt.show()
