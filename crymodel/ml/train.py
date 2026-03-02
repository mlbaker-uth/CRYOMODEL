# crymodel/ml/train.py
"""Training script for ion/water classification model."""
from __future__ import annotations
import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .model import IonWaterMLP, focal_ce

ION_CLASSES = ["HOH", "Na", "K", "Mg", "Ca", "Mn", "Fe2", "Fe3", "Cl", "Zn"]
ION_SET = set(ION_CLASSES)
WATER_LABEL = "HOH"


def label_to_idx(y: np.ndarray) -> np.ndarray:
    """Convert label strings to class indices."""
    return np.array([ION_CLASSES.index(v) for v in y], dtype=np.int64)


class FeatDataset(Dataset):
    """PyTorch dataset for feature arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        if self.y is None:
            return self.X[i]
        return self.X[i], self.y[i]


def main(
    train_csv: str | None = None,
    outdir: str = "ionwater_env_model",
    epochs: int = 40,
    batch: int = 512,
    lr: float = 2e-4,
    focal: bool = False,
    class_weights: bool = False,
    group_col: str = "pdb_id",
):
    """Train ion/water classification model.
    
    Can be called directly with parameters or via argparse (for CLI).
    """
    # Support argparse for backward compatibility
    if train_csv is None:
        ap = argparse.ArgumentParser(description="Train ion/water classification model")
        ap.add_argument("--train-csv", required=True, help="Features CSV with labels")
        ap.add_argument("--outdir", default="ionwater_env_model", help="Output directory for model")
        ap.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
        ap.add_argument("--batch", type=int, default=512, help="Batch size")
        ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
        ap.add_argument("--focal", action="store_true", help="Use focal loss")
        ap.add_argument("--class-weights", action="store_true", help="Use class weights")
        ap.add_argument("--group-col", default="pdb_id", help="Column for group-based splitting (to avoid leakage)")
        args = ap.parse_args()
        train_csv = args.train_csv
        outdir = args.outdir
        epochs = args.epochs
        batch = args.batch
        lr = args.lr
        focal = args.focal
        class_weights = args.class_weights
        group_col = args.group_col

    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(train_csv)
    df = df[df["label"].isin(ION_SET)].copy()

    # Gather feature columns (everything numeric except coordinates/ids/label)
    drop_cols = {"label", "x", "y", "z", "group_id", "pdb_id", "resname", "chain", "resi"}
    feat_cols = [c for c in df.columns if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]
    X = df[feat_cols].to_numpy()
    y = label_to_idx(df["label"].to_numpy())

    # Replace inf/nan BEFORE scaling (StandardScaler can't handle inf)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Scale
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # Replace inf/nan again after scaling (shouldn't happen, but safety check)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=1e6, neginf=-1e6)

    # Class weights (optional)
    cw = None
    if class_weights:
        counts = np.bincount(y, minlength=len(ION_CLASSES))
        inv = 1.0 / np.clip(counts, 1, None)
        cw = torch.tensor(inv / np.sum(inv) * len(ION_CLASSES), dtype=torch.float32)

    # Split by group (e.g., pdb_id)
    groups = df[group_col].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=5)
    tr_idx, val_idx = next(gkf.split(Xs, y, groups=groups))

    Xtr, ytr = Xs[tr_idx], y[tr_idx]
    Xva, yva = Xs[val_idx], y[val_idx]

    ds_tr = FeatDataset(Xtr, ytr)
    ds_va = FeatDataset(Xva, yva)

    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IonWaterMLP(in_dim=Xs.shape[1], n_classes=len(ION_CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            logits, aux = model(xb, calibrate=False)
            # main loss
            if focal:
                loss_main = focal_ce(logits, yb, weight=cw.to(device) if cw is not None else None, gamma=1.5)
            else:
                loss_main = nn.CrossEntropyLoss(weight=cw.to(device) if cw is not None else None)(logits, yb)
            # aux: water vs ion
            y_aux = (yb != ION_CLASSES.index(WATER_LABEL)).long()
            loss_aux = nn.CrossEntropyLoss()(aux, y_aux)
            loss = loss_main + 0.3 * loss_aux

            opt.zero_grad()
            loss.backward()
            opt.step()

        # validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                logits, _ = model(xb, calibrate=False)
                pred = logits.argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        acc = correct / total if total else 0.0
        if acc > best_val:
            best_val = acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feat_cols": feat_cols,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "classes": ION_CLASSES,
                },
                os.path.join(outdir, "model.pt"),
            )
        print(f"epoch {epoch:02d}  val_acc={acc:.3f}  best={best_val:.3f}")

    # Temperature scaling on val set
    ckpt = torch.load(os.path.join(outdir, "model.pt"), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    log_temp = model.log_temp
    log_temp.requires_grad_(True)
    opt_t = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50)

    xb = torch.tensor(Xva, dtype=torch.float32, device=device)
    yb = torch.tensor(yva, dtype=torch.long, device=device)

    def closure():
        opt_t.zero_grad()
        logits, _ = model(xb, calibrate=True)  # uses log_temp
        nll = nn.CrossEntropyLoss()(logits, yb)
        nll.backward()
        return nll

    opt_t.step(closure)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feat_cols": feat_cols,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "classes": ION_CLASSES,
        },
        os.path.join(outdir, "model.pt"),
    )
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump({"best_val_acc": best_val, "classes": ION_CLASSES}, f, indent=2)
    print(f"Training complete. Best validation accuracy: {best_val:.3f}")


if __name__ == "__main__":
    main()

