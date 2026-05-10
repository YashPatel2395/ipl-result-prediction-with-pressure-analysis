"""
_seq_worker.py
==============
Subprocess worker for LSTM/GRU sequence model training.
Called by sequence_model.py via subprocess.run.
Receives numpy arrays via .npy files, writes results to CSV.

Usage (internal):
    python -m src._seq_worker <X_tr.npy> <y_tr.npy> <X_te.npy> <y_te.npy> <n_feat> <out_csv>
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

def main():
    X_tr_path, y_tr_path, X_te_path, y_te_path, n_feat_str, out_csv = sys.argv[1:]
    n_feat = int(n_feat_str)

    X_tr = np.load(X_tr_path).astype(np.float32)
    y_tr = np.load(y_tr_path).astype(np.float32)
    X_te = np.load(X_te_path).astype(np.float32)
    y_te = np.load(y_te_path).astype(np.float32)

    print(f"  [worker] X_tr: {X_tr.shape}  X_te: {X_te.shape}", flush=True)

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        torch_ok = True
    except ImportError:
        torch_ok = False

    records = []

    if torch_ok:
        for arch in ["LSTM", "GRU"]:
            print(f"  [worker] Training {arch}...", flush=True)
            try:
                class _Net(nn.Module):
                    def __init__(self, inp, hid=64, layers=2, drop=0.3):
                        super().__init__()
                        rnn_cls = nn.LSTM if arch == "LSTM" else nn.GRU
                        self.rnn = rnn_cls(inp, hid, layers, batch_first=True,
                                           dropout=drop if layers > 1 else 0)
                        self.fc = nn.Sequential(
                            nn.Linear(hid, 32), nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(32, 1), nn.Sigmoid(),
                        )
                    def forward(self, x):
                        out = self.rnn(x)
                        h = out[1][0] if arch == "LSTM" else out[1]
                        return self.fc(h[-1]).squeeze(-1)

                pos = int(y_tr.sum()); neg = len(y_tr) - pos
                w_pos = neg / (pos + 1e-9)

                model = _Net(n_feat)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
                loss_fn = torch.nn.BCELoss(reduction="none")

                X_t = torch.from_numpy(X_tr)
                y_t = torch.from_numpy(y_tr.copy())
                dl = DataLoader(TensorDataset(X_t, y_t), batch_size=1024, shuffle=True)

                model.train()
                for epoch in range(15):
                    for xb, yb in dl:
                        opt.zero_grad()
                        pred = model(xb)
                        w = torch.where(yb == 1,
                                        torch.full_like(yb, w_pos),
                                        torch.ones_like(yb))
                        loss = (loss_fn(pred, yb) * w).mean()
                        loss.backward()
                        opt.step()
                    if (epoch + 1) % 5 == 0:
                        model.eval()
                        with torch.no_grad():
                            vp = model(torch.from_numpy(X_te)).numpy()
                        val_auc = roc_auc_score(y_te, vp)
                        print(f"    Epoch {epoch+1}/15  val_AUC={val_auc:.4f}", flush=True)
                        model.train()

                model.eval()
                with torch.no_grad():
                    y_proba = model(torch.from_numpy(X_te)).numpy()
                y_pred = (y_proba >= 0.5).astype(int)
                records.append({
                    "model":   arch,
                    "roc_auc": round(roc_auc_score(y_te, y_proba), 4),
                    "f1":      round(f1_score(y_te, y_pred),        4),
                    "brier":   round(brier_score_loss(y_te, y_proba), 5),
                })
                print(f"  [worker] {arch} done: AUC={records[-1]['roc_auc']:.4f}", flush=True)
            except Exception as e:
                print(f"  [worker] {arch} failed: {e}", flush=True)

    if not records:
        # MLP fallback
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        print("  [worker] MLP fallback...", flush=True)
        X_tr_f = X_tr.reshape(len(X_tr), -1)
        X_te_f = X_te.reshape(len(X_te), -1)
        sc = StandardScaler()
        X_tr_f = sc.fit_transform(X_tr_f)
        X_te_f = sc.transform(X_te_f)
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=30,
                            random_state=42, early_stopping=True,
                            validation_fraction=0.1, n_iter_no_change=5)
        mlp.fit(X_tr_f, y_tr)
        y_proba = mlp.predict_proba(X_te_f)[:, 1]
        y_pred = mlp.predict(X_te_f)
        records.append({
            "model":   "MLP (flattened seq, fallback)",
            "roc_auc": round(roc_auc_score(y_te, y_proba), 4),
            "f1":      round(f1_score(y_te, y_pred),        4),
            "brier":   round(brier_score_loss(y_te, y_proba), 5),
        })
        print(f"  [worker] MLP done: AUC={records[-1]['roc_auc']:.4f}", flush=True)

    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"  [worker] Results saved → {out_csv}", flush=True)

if __name__ == "__main__":
    main()
