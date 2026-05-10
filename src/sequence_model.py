"""
sequence_model.py
=================
Sequence-aware chase prediction using LSTM and GRU networks.

Rather than treating each ball as an independent observation (tabular approach),
sequential models see the last SEQ_LEN deliveries as a temporal context window
and learn patterns in the ORDER of events — e.g., a wicket followed by dot balls
followed by another wicket is a collapse pattern not visible from a single snapshot.

Architecture
------------
  Input:  (batch, SEQ_LEN, n_features) — last N ball-states
  LSTM/GRU: hidden_size=64, num_layers=2, dropout=0.3
  FC head: 64 → 32 → ReLU → Dropout → 1 → Sigmoid
  Loss:   BCELoss with class-weight correction
  Optim:  Adam (lr=1e-3, weight_decay=1e-4)

Evaluation
----------
  Compared against XGBoost (Config C) on the same held-out matches.
  Metric: ROC-AUC, F1, Brier Score.

Note: requires PyTorch. If unavailable, an MLP on flattened sequences
(sklearn) is used as a fallback with a clear warning.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "plots")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")

SEQ_LEN = 20   # context window: last 20 deliveries (~3.3 overs)

# Core features for sequence model (avoid redundancy; momentum-aware)
SEQ_FEATURES = [
    "current_score", "runs_remaining", "balls_remaining",
    "wickets_lost", "wickets_in_hand",
    "current_run_rate", "required_run_rate",
    "pressure", "dp_momentum", "dp_ewm_fast",
    "runs_last_6", "dot_balls_last_12", "scoring_acceleration",
]


# ===========================================================================
# Sequence construction
# ===========================================================================

def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    For each ball, build a (seq_len × n_feat) context window using
    the PREVIOUS seq_len deliveries (padded with zeros if < seq_len).

    Returns:
      seqs   — (n_samples, seq_len, n_feat)
      labels — (n_samples,)
      indices — original df row indices (for train/test split)
    """
    valid_cols = [c for c in feature_cols if c in df.columns]
    all_seqs, all_labels, all_indices = [], [], []

    for match_id, grp in df.groupby("match_id", sort=False):
        grp = grp.sort_values("balls_bowled").reset_index()
        X   = grp[valid_cols].values.astype(np.float32)
        y   = grp["won"].values
        idx = grp["index"].values

        n     = len(X)
        # Zero-pad prefix so every ball has a full-length context
        X_pad = np.vstack([np.zeros((seq_len - 1, len(valid_cols)), dtype=np.float32), X])

        for i in range(n):
            seq = X_pad[i: i + seq_len]   # (seq_len, n_feat) — past-only
            all_seqs.append(seq)
            all_labels.append(int(y[i]))
            all_indices.append(int(idx[i]))

    return np.array(all_seqs, dtype=np.float32), \
           np.array(all_labels, dtype=np.float32), \
           all_indices


# ===========================================================================
# PyTorch models
# ===========================================================================

def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        return torch, nn, TensorDataset, DataLoader
    except ImportError:
        return None, None, None, None


class LSTMPredictor:
    """Wrapper around PyTorch LSTM for sklearn-style interface."""

    def __init__(self, input_size: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.3, lr: float = 1e-3, epochs: int = 15,
                 batch_size: int = 512):
        self.input_size = input_size
        self.hidden     = hidden
        self.layers     = layers
        self.dropout    = dropout
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.model_     = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        torch, nn, TensorDataset, DataLoader = _try_import_torch()
        if torch is None:
            raise ImportError("PyTorch not available")

        class _LSTM(nn.Module):
            def __init__(self, inp, hid, layers, drop):
                super().__init__()
                self.rnn = nn.LSTM(inp, hid, layers, batch_first=True,
                                   dropout=drop if layers > 1 else 0)
                self.fc  = nn.Sequential(
                    nn.Linear(hid, 32), nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1),  nn.Sigmoid(),
                )
            def forward(self, x):
                _, (h, _) = self.rnn(x)
                return self.fc(h[-1]).squeeze(-1)

        pos   = int(y.sum());  neg = len(y) - pos
        w_pos = neg / (pos + 1e-9)

        model  = _LSTM(self.input_size, self.hidden, self.layers, self.dropout)
        opt    = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = torch.nn.BCELoss(reduction="none")

        X_t = torch.from_numpy(X)          # zero-copy view of numpy array
        y_t = torch.from_numpy(y.copy())   # copy needed for shuffle safety
        ds  = TensorDataset(X_t, y_t)
        dl  = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        model.train()
        for epoch in range(self.epochs):
            ep_loss = 0.0
            for xb, yb in dl:
                opt.zero_grad()
                pred = model(xb)
                w    = torch.where(yb == 1,
                                   torch.full_like(yb, w_pos),
                                   torch.ones_like(yb))
                loss = (loss_fn(pred, yb) * w).mean()
                loss.backward()
                opt.step()
                ep_loss += loss.item()

            if X_val is not None and (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    vp = model(torch.tensor(X_val)).numpy()
                val_auc = roc_auc_score(y_val, vp)
                print(f"    Epoch {epoch+1}/{self.epochs}  "
                      f"loss={ep_loss/len(dl):.4f}  val_AUC={val_auc:.4f}")
                model.train()

        self.model_ = model
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch, *_ = _try_import_torch()
        if torch is None or self.model_ is None:
            raise RuntimeError("Model not trained")
        self.model_.eval()
        with torch.no_grad():
            proba = self.model_(torch.from_numpy(X)).numpy()
        return np.column_stack([1 - proba, proba])


class GRUPredictor(LSTMPredictor):
    """GRU variant — same interface as LSTM."""

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        torch, nn, TensorDataset, DataLoader = _try_import_torch()
        if torch is None:
            raise ImportError("PyTorch not available")

        class _GRU(nn.Module):
            def __init__(self, inp, hid, layers, drop):
                super().__init__()
                self.rnn = nn.GRU(inp, hid, layers, batch_first=True,
                                  dropout=drop if layers > 1 else 0)
                self.fc  = nn.Sequential(
                    nn.Linear(hid, 32), nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1),  nn.Sigmoid(),
                )
            def forward(self, x):
                _, h = self.rnn(x)
                return self.fc(h[-1]).squeeze(-1)

        pos = int(y.sum());  neg = len(y) - pos
        w_pos = neg / (pos + 1e-9)

        model   = _GRU(self.input_size, self.hidden, self.layers, self.dropout)
        opt     = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = torch.nn.BCELoss(reduction="none")

        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y.copy())
        ds  = TensorDataset(X_t, y_t)
        dl  = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        model.train()
        for epoch in range(self.epochs):
            ep_loss = 0.0
            for xb, yb in dl:
                opt.zero_grad()
                pred = model(xb)
                w    = torch.where(yb == 1,
                                   torch.full_like(yb, w_pos),
                                   torch.ones_like(yb))
                loss = (loss_fn(pred, yb) * w).mean()
                loss.backward()
                opt.step()
                ep_loss += loss.item()

            if X_val is not None and (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    vp = model(torch.tensor(X_val)).numpy()
                val_auc = roc_auc_score(y_val, vp)
                print(f"    Epoch {epoch+1}/{self.epochs}  "
                      f"loss={ep_loss/len(dl):.4f}  val_AUC={val_auc:.4f}")
                model.train()

        self.model_ = model
        return self


# ===========================================================================
# Fallback: MLP on flattened sequences (sklearn)
# ===========================================================================

def _mlp_fallback(X_tr, y_tr, X_te, y_te):
    from sklearn.neural_network import MLPClassifier
    X_tr_flat = X_tr.reshape(len(X_tr), -1)
    X_te_flat = X_te.reshape(len(X_te), -1)

    scaler     = StandardScaler()
    X_tr_s     = scaler.fit_transform(X_tr_flat)
    X_te_s     = scaler.transform(X_te_flat)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=30,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        verbose=False,
    )
    mlp.fit(X_tr_s, y_tr)
    y_proba = mlp.predict_proba(X_te_s)[:, 1]
    y_pred  = mlp.predict(X_te_s)
    return {
        "model":    "MLP (flattened seq, sklearn fallback)",
        "roc_auc":  round(roc_auc_score(y_te, y_proba),   4),
        "f1":       round(f1_score(y_te, y_pred),          4),
        "brier":    round(brier_score_loss(y_te, y_proba), 5),
    }, y_proba


# ===========================================================================
# Main training and comparison
# ===========================================================================

def run_sequence_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build sequences, train LSTM + GRU (or MLP fallback), compare to XGBoost.
    """
    valid_cols = [c for c in SEQ_FEATURES if c in df.columns]
    print(f"\n[sequence_model] Building {SEQ_LEN}-ball context sequences "
          f"on {len(valid_cols)} features...")

    seqs, labels, indices = build_sequences(df, valid_cols, SEQ_LEN)
    print(f"  Sequence dataset shape: {seqs.shape}  labels: {labels.shape}")

    # Match-level train/test split using the same indices
    idx_arr       = np.array(indices)
    match_of_ball = df["match_id"].values
    match_labels  = df.groupby("match_id")["won"].first()
    all_matches   = df["match_id"].unique()

    train_matches, test_matches = train_test_split(
        all_matches, test_size=0.2, random_state=42,
        stratify=match_labels[all_matches],
    )
    train_match_set = set(train_matches)
    test_match_set  = set(test_matches)

    # Map each ball's original index → its match_id → train/test
    ball_match = df["match_id"].to_dict()   # index → match_id (fast vectorised)
    tr_mask = np.array([ball_match.get(i, -1) in train_match_set for i in idx_arr])
    te_mask = np.array([ball_match.get(i, -1) in test_match_set  for i in idx_arr])

    X_tr, y_tr = seqs[tr_mask], labels[tr_mask]
    X_te, y_te = seqs[te_mask], labels[te_mask]
    del seqs, labels   # free full array — we only need train/test splits
    import gc; gc.collect()
    print(f"  Train: {X_tr.shape[0]:,} balls | Test: {X_te.shape[0]:,} balls")

    # Subsample train to 40k to keep memory under control for LSTM
    MAX_TRAIN = 40_000
    if len(X_tr) > MAX_TRAIN:
        rng  = np.random.default_rng(42)
        idx  = rng.choice(len(X_tr), MAX_TRAIN, replace=False)
        X_tr = X_tr[idx]
        y_tr = y_tr[idx]
        print(f"  Subsampled train to {MAX_TRAIN:,} for sequence models (test unchanged)")

    # Normalise per-feature (across time and samples)
    scaler   = StandardScaler()
    X_tr_2d  = X_tr.reshape(-1, X_tr.shape[-1])
    X_te_2d  = X_te.reshape(-1, X_te.shape[-1])
    scaler.fit(X_tr_2d)
    X_tr = scaler.transform(X_tr_2d).reshape(X_tr.shape).astype(np.float32)
    del X_tr_2d
    X_te = scaler.transform(X_te_2d).reshape(X_te.shape).astype(np.float32)
    del X_te_2d
    gc.collect()

    # Run LSTM/GRU training in an isolated subprocess to avoid memory conflicts
    # with the large models from Step 21 still in the parent process.
    import subprocess, tempfile, sys as _sys

    results = []
    tmp_dir = tempfile.mkdtemp()
    np.save(os.path.join(tmp_dir, "X_tr.npy"), X_tr)
    np.save(os.path.join(tmp_dir, "y_tr.npy"), y_tr)
    np.save(os.path.join(tmp_dir, "X_te.npy"), X_te)
    np.save(os.path.join(tmp_dir, "y_te.npy"), y_te)
    out_csv = os.path.join(tmp_dir, "seq_results.csv")

    print("\n  [sequence_model] Launching isolated subprocess for LSTM/GRU...", flush=True)
    proc = subprocess.run(
        [_sys.executable, "-m", "src._seq_worker",
         os.path.join(tmp_dir, "X_tr.npy"),
         os.path.join(tmp_dir, "y_tr.npy"),
         os.path.join(tmp_dir, "X_te.npy"),
         os.path.join(tmp_dir, "y_te.npy"),
         str(len(valid_cols)),
         out_csv],
        capture_output=False,   # stream stdout/stderr to terminal
        timeout=600,            # 10-minute cap
        cwd=PROJECT_ROOT,
    )

    if proc.returncode == 0 and os.path.exists(out_csv):
        seq_df = pd.read_csv(out_csv)
        for _, row in seq_df.iterrows():
            results.append(row.to_dict())
            print(f"  {row['model']}: AUC={row['roc_auc']:.4f}  "
                  f"F1={row['f1']:.4f}  Brier={row['brier']:.5f}")
    else:
        print(f"  [sequence_model] Subprocess exited {proc.returncode} — "
              f"falling back to MLP", flush=True)
        m, _ = _mlp_fallback(X_tr, y_tr, X_te, y_te)
        results.append(m)
        print(f"  MLP: AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}  Brier={m['brier']:.5f}")

    results_df = pd.DataFrame(results)

    _plot_sequence_comparison(results_df)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(OUTPUTS_DIR, "sequence_model_results.csv"), index=False)
    return results_df


def _plot_sequence_comparison(results: pd.DataFrame) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = ["roc_auc", "f1", "brier"]
    titles  = ["ROC-AUC", "F1 Score", "Brier Score (↓)"]
    palette = sns_palette = ["#E67E22", "#9B59B6", "#1ABC9C"]

    try:
        import seaborn as sns
        for ax, metric, title, color in zip(axes, metrics, titles, palette):
            bars = ax.bar(results["model"], results[metric], color=color, alpha=0.85, width=0.4)
            for bar, val in zip(bars, results[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
            ax.set_title(f"Sequential Models — {title}", fontweight="bold")
            ax.set_ylabel(title)
    except Exception:
        for ax, metric, title in zip(axes, metrics, titles):
            ax.bar(results["model"], results[metric], alpha=0.8)
            ax.set_title(f"Sequential Models — {title}", fontweight="bold")

    fig.suptitle(f"Sequence Model Comparison (context window = {SEQ_LEN} balls)\n"
                 "LSTM vs GRU on ball-by-ball chase state sequences",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "33_sequence_model_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [sequence_model] Saved → {path}")
