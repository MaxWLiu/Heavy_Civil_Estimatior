#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import bid_feature_extractor as bfe

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Network parameters
net_params = {
    'input_size': 1,
    'hidden1': 10,
    'hidden2': 20,
    'hidden3': 10,
    'output_size': 1,
    'dropout': 0.2
}
trainer_info = {
    'epochs': 100,
    'batch_size': 16,
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'test_size': 0.2
}

# Define a simple 3-layer NN
class ThreeLayerNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        dr = params['dropout']
        self.net = nn.Sequential(
            nn.Linear(params['input_size'], params['hidden1']),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(params['hidden1'], params['hidden2']),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(params['hidden2'], params['hidden3']),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(params['hidden3'], params['output_size'])
        )
    def forward(self, x):
        return self.net(x)

# Main routine
def main():
    # 1. Load features (only quantity)
    _, unit, qty, _, _, _ = bfe.extract_bid_features("840515", 1, 3)
    log_unit = np.log(np.where(unit > 0, unit, np.nan))
    log_qty  = np.log(np.where(qty  > 0,  qty,  np.nan))
    mask = np.isfinite(log_unit) & np.isfinite(log_qty)
    X_np = log_qty[mask].reshape(-1,1)
    Y_np = log_unit[mask].reshape(-1,1)

    # 2. Scale
    sx, sy = StandardScaler(), StandardScaler()
    X_scaled = sx.fit_transform(X_np)
    Y_scaled = sy.fit_transform(Y_np)

    # 3. Train/test split
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_scaled, Y_scaled,
        test_size=trainer_info['test_size'], random_state=42
    )

    # Convert to torch DataLoaders
    def make_loader(X, Y):
        tX = torch.from_numpy(X).float().to(device)
        tY = torch.from_numpy(Y).float().to(device)
        ds = torch.utils.data.TensorDataset(tX, tY)
        return torch.utils.data.DataLoader(ds, batch_size=trainer_info['batch_size'], shuffle=True)
    train_loader = make_loader(X_tr, Y_tr)
    test_loader  = make_loader(X_te, Y_te)

    # 4. Initialize model, loss, optimizer
    model = ThreeLayerNet(net_params).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=trainer_info['lr'], weight_decay=trainer_info['weight_decay'])

    # 5. Training with convergence tracking
    epochs = trainer_info['epochs']
    train_losses = np.zeros(epochs)
    test_losses  = np.zeros(epochs)

    for ep in range(1, epochs+1):
        # train step
        model.train()
        total_train = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_train += loss.item() * xb.size(0)
        train_losses[ep-1] = total_train / len(train_loader.dataset)

        # test step
        model.eval()
        total_test = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb)
                total_test += criterion(preds, yb).item() * xb.size(0)
        test_losses[ep-1] = total_test / len(test_loader.dataset)

        if ep % 10 == 0:
            print(f"Epoch {ep}/{epochs} - Train Loss: {train_losses[ep-1]:.4f}, Test Loss: {test_losses[ep-1]:.4f}")

    # 6. Retrain on full data for fitting
    full_loader = make_loader(X_scaled, Y_scaled)
    final_model = ThreeLayerNet(net_params).to(device)
    opt_f = optim.Adam(final_model.parameters(), lr=trainer_info['lr'], weight_decay=trainer_info['weight_decay'])
    final_model.train()
    for _ in range(epochs):
        for xb, yb in full_loader:
            opt_f.zero_grad()
            loss = criterion(final_model(xb), yb)
            loss.backward()
            opt_f.step()

    # 7. Generate fit line
    grid = np.linspace(X_np.min(), X_np.max(), 200).reshape(-1,1)
    grid_scaled = sx.transform(grid)
    final_model.eval()
    with torch.no_grad():
        y_pred_scaled = final_model(torch.from_numpy(grid_scaled).float().to(device)).cpu().numpy()
    grid_orig = grid
    y_pred_orig = sy.inverse_transform(y_pred_scaled)

    # 8. Plot: fit and convergence as subplots
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    # Fit
    axes[0].scatter(X_np, Y_np, alpha=0.3)
    axes[0].plot(grid_orig, y_pred_orig, color='orange')
    axes[0].set_xlabel('log(Qty)'); axes[0].set_ylabel('log(Unit Cost)')
    axes[0].set_title('NN Regression Fit'); axes[0].grid(True)
    # Convergence
    axes[1].plot(np.arange(1, epochs+1), train_losses, label='Train')
    axes[1].plot(np.arange(1, epochs+1), test_losses,  label='Test', linestyle='--')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MSE Loss')
    axes[1].set_yscale('log'); axes[1].set_title('Convergence'); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    main()
