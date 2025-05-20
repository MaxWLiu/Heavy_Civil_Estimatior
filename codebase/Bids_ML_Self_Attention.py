import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt

from Bid_Item_ML_Pipeline import DB_ROOT, TimeSeriesFactorDB

# Paths
BIDS_DB = DB_ROOT / "bid_database.db"
TF_DB   = DB_ROOT / "time_adjustment_factors.db"

# Output directory for model and plots
default_save_dir = Path(r"D:/Estimation Code")
default_save_dir.mkdir(parents=True, exist_ok=True)

PAD_IDX = 0
EPS = 1e-6

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MultiItemBidDataset(Dataset):
    """
    Dataset for bids: uses log-transformed target and normalized features with CCCI-adjusted unit prices.
    """
    def __init__(self, bids_db_path: str):
        self.time_db = TimeSeriesFactorDB(str(TF_DB))
        norm_date = datetime(2024, 12, 1)
        self.target_ccci = self.time_db.get_ccci(norm_date.year, norm_date.month)

        conn = sqlite3.connect(bids_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT Contract_Number, Bid_Rank, Item_Number, Qty, Unit_Price, Total, UOM, Bid_Open_Date
            FROM bids
        """)
        rows = cursor.fetchall()
        conn.close()

        tmp: Dict[Tuple[str,int], List[Tuple]] = {}
        for c, r, itm, qty, up, tot, uom, date_str in rows:
            if r is None: continue
            key = (c, int(r))
            tmp.setdefault(key, []).append((int(itm), float(qty), float(up), float(tot), uom=='LS', date_str))
        self.bids = [(c, r, items) for (c, r), items in tmp.items()]

        all_ids = {itm for _, _, items in self.bids for itm, *_ in items}
        self.id2idx = {itm: idx+1 for idx, itm in enumerate(sorted(all_ids))}
        self.vocab_size = len(self.id2idx) + 1

        # Precompute stats
        qty_list, uc_list, rank_list, target_list = [], [], [], []
        for _, r, items in self.bids:
            rank_list.append(r)
            total_ls = 0.0
            for itm, qty, up, tot, is_ls, date_str in items:
                qty_list.append(qty)
                d = datetime.fromisoformat(date_str)
                y, m = (d.year, d.month % 12 + 1)
                ccci = self.time_db.get_ccci(y, m)
                factor = self.target_ccci / ccci if ccci>0 else 1.0
                adj_up = up * factor
                uc_list.append(adj_up)
                if is_ls: total_ls += tot
            target_list.append(np.log1p(total_ls))
        self.q_mean, self.q_std = np.mean(qty_list), np.std(qty_list)+EPS
        self.uc_mean, self.uc_std = np.mean(uc_list), np.std(uc_list)+EPS
        self.rank_mean, self.rank_std = np.mean(rank_list), np.std(rank_list)+EPS
        self.t_mean, self.t_std = np.mean(target_list), np.std(target_list)+EPS

    def __len__(self): return len(self.bids)

    def __getitem__(self, idx: int):
        _, rank, items = self.bids[idx]
        token_list, total_ls = [], 0.0
        for itm, qty, up, tot, is_ls, date_str in items:
            iid = self.id2idx[itm]
            qty_n = (qty - self.q_mean)/self.q_std
            d = datetime.fromisoformat(date_str)
            y, m = (d.year, d.month % 12 + 1)
            ccci = self.time_db.get_ccci(y, m)
            factor = self.target_ccci/ccci if ccci>0 else 1.0
            adj_up = up*factor
            up_n = (adj_up-self.uc_mean)/self.uc_std
            token_list.append((iid, qty_n, up_n))
            if is_ls: total_ls += tot
        rank_n = (rank - self.rank_mean)/self.rank_std
        target_n = (np.log1p(total_ls)-self.t_mean)/self.t_std
        item_ids = np.array([t[0] for t in token_list], int)
        quantities = np.array([t[1] for t in token_list], float)
        unit_costs = np.array([t[2] for t in token_list], float)
        return {'item_ids':item_ids,'quantities':quantities,'unit_costs':unit_costs,'mask':None,'ranks':np.array([rank_n]),'target':np.array([target_n])}

    @staticmethod
    def collate_fn(batch):
        max_len = max(len(d['item_ids']) for d in batch)
        B = len(batch)
        item_ids = torch.full((B,max_len),PAD_IDX,dtype=torch.long)
        quantities = torch.zeros((B,max_len))
        unit_costs = torch.zeros((B,max_len))
        mask = torch.zeros((B,max_len),dtype=torch.bool)
        ranks = torch.zeros((B,1))
        targets = torch.zeros((B,1))
        for i,d in enumerate(batch):
            L = len(d['item_ids'])
            item_ids[i,:L]=torch.from_numpy(d['item_ids'])
            quantities[i,:L]=torch.from_numpy(d['quantities'])
            unit_costs[i,:L]=torch.from_numpy(d['unit_costs'])
            mask[i,:L]=True
            ranks[i]=torch.from_numpy(d['ranks'])
            targets[i]=torch.from_numpy(d['target'])
        return {'item_ids':item_ids.to(device),'quantities':quantities.to(device),'unit_costs':unit_costs.to(device),'mask':mask.to(device),'ranks':ranks.to(device),'target':targets.to(device)}

class MultiItemTransformer(nn.Module):
    def __init__(self,vocab_size,embed_dim=128,nhead=8,num_layers=4,dim_feedforward=256,dropout=0.1):
        super().__init__()
        self.item_embed=nn.Embedding(vocab_size,embed_dim,padding_idx=PAD_IDX)
        self.feature_proj=nn.Sequential(nn.Linear(2,embed_dim),nn.ReLU(),nn.Linear(embed_dim,embed_dim))
        self.rank_proj=nn.Linear(1,embed_dim)
        enc_layer=nn.TransformerEncoderLayer(d_model=embed_dim,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout,batch_first=True)
        self.transformer=nn.TransformerEncoder(enc_layer,num_layers=num_layers)
        self.regressor=nn.Sequential(nn.Dropout(dropout),nn.Linear(embed_dim,embed_dim//2),nn.ReLU(),nn.Dropout(dropout),nn.Linear(embed_dim//2,1))
    def forward(self,item_ids,quantities,unit_costs,mask,ranks):
        x=self.item_embed(item_ids)
        f=torch.stack([quantities,unit_costs],dim=-1)
        x=x+self.feature_proj(f)
        enc=self.transformer(x,src_key_padding_mask=~mask)
        m=mask.unsqueeze(-1)
        pooled=(enc*m).sum(1)/m.sum(1).clamp(min=1)
        pooled=pooled+self.rank_proj(ranks)
        return self.regressor(pooled)


def train_transformer(num_epochs=100,batch_size=32,lr=5e-4,test_split=0.2,patience=10):
    dataset=MultiItemBidDataset(str(BIDS_DB))
    test_size=int(len(dataset)*test_split);train_size=len(dataset)-test_size
    train_ds,test_ds=random_split(dataset,[train_size,test_size])
    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,collate_fn=MultiItemBidDataset.collate_fn)
    test_loader=DataLoader(test_ds,batch_size=batch_size,shuffle=False,collate_fn=MultiItemBidDataset.collate_fn)

    model=MultiItemTransformer(dataset.vocab_size).to(device)
    opt=optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-2)
    scheduler=StepLR(opt,step_size=20,gamma=0.7)
    criterion=nn.SmoothL1Loss()  # Huber loss reduces sensitivity to outliers
    best_loss=float('inf');epochs_no_improve=0

    train_losses,test_losses=[],[]
    for epoch in range(1,num_epochs+1):
        model.train();total_train=0
        for batch in train_loader:
            preds=model(batch['item_ids'],batch['quantities'],batch['unit_costs'],batch['mask'],batch['ranks'])
            loss=criterion(preds,batch['target'])
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.0);opt.step()
            total_train+=loss.item()*preds.size(0)
        avg_train=total_train/train_size;train_losses.append(avg_train)

        model.eval();total_test=0
        with torch.no_grad():
            for batch in test_loader:
                preds=model(batch['item_ids'],batch['quantities'],batch['unit_costs'],batch['mask'],batch['ranks'])
                total_test+=criterion(preds,batch['target']).item()*preds.size(0)
        avg_test=total_test/test_size;test_losses.append(avg_test)
        print(f"Epoch {epoch}/{num_epochs} — Train Loss: {avg_train:.4f} — Test Loss: {avg_test:.4f}")

        # early stopping
        if avg_test<best_loss:
            best_loss=avg_test;epochs_no_improve=0
        else:
            epochs_no_improve+=1
            if epochs_no_improve>=patience:
                print(f"No improvement {patience} epochs, stopping.");break
        scheduler.step()

    torch.save(model.state_dict(),str(default_save_dir/"multi_item_transformer.pt"))
    plt.figure();plt.plot(train_losses,label='Train');plt.plot(test_losses,label='Test');plt.xlabel('Epoch');plt.ylabel('Loss');plt.legend();plt.grid(True)
    plt.savefig(str(default_save_dir/"loss_curve.png"))
    print("Saved model and loss curve.")

if __name__=="__main__": train_transformer()
