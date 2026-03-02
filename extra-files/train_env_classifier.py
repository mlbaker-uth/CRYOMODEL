#!/usr/bin/env python3
import argparse, os, json
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import IonWaterMLP, focal_ce

ION_CLASSES = ["HOH","Na","K","Mg","Ca","Mn","Fe2","Fe3","Cl","Zn"]
ION_SET = set(ION_CLASSES)
WATER_LABEL = "HOH"

def label_to_idx(y):
    return np.array([ION_CLASSES.index(v) for v in y], dtype=np.int64)

class FeatDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        if self.y is None:
            return self.X[i]
        return self.X[i], self.y[i]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True, help="features CSV with labels")
    ap.add_argument("--outdir", default="ionwater_env_model")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--focal", action="store_true")
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--group-col", default="pdb_id", help="to avoid leakage across splits")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.train_csv)
    df = df[df["label"].isin(ION_SET)].copy()

    # gather feature columns (everything numeric except coordinates/ids/label)
    drop_cols = {"label","x","y","z","group_id","pdb_id"}
    feat_cols = [c for c in df.columns if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]
    X = df[feat_cols].to_numpy()
    y = label_to_idx(df["label"].to_numpy())

    # scale
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # class weights (optional)
    cw = None
    if args.class_weights:
        counts = np.bincount(y, minlength=len(ION_CLASSES))
        inv = 1.0 / np.clip(counts, 1, None)
        cw = torch.tensor(inv/np.sum(inv)*len(ION_CLASSES), dtype=torch.float32)

    # split by group (e.g., pdb_id)
    groups = df[args.group_col].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=5)
    tr_idx, val_idx = next(gkf.split(Xs, y, groups=groups))

    Xtr, ytr = Xs[tr_idx], y[tr_idx]
    Xva, yva = Xs[val_idx], y[val_idx]

    ds_tr = FeatDataset(Xtr, ytr)
    ds_va = FeatDataset(Xva, yva)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IonWaterMLP(in_dim=Xs.shape[1], n_classes=len(ION_CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            logits, aux = model(xb, calibrate=False)
            # main loss
            if args.focal:
                loss_main = focal_ce(logits, yb, weight=cw.to(device) if cw is not None else None, gamma=1.5)
            else:
                loss_main = nn.CrossEntropyLoss(weight=cw.to(device) if cw is not None else None)(logits, yb)
            # aux: water vs ion
            y_aux = (yb != ION_CLASSES.index(WATER_LABEL)).long()
            loss_aux = nn.CrossEntropyLoss()(aux, y_aux)
            loss = loss_main + 0.3*loss_aux

            opt.zero_grad(); loss.backward(); opt.step()

        # validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                logits, _ = model(xb, calibrate=False)
                pred = logits.argmax(dim=-1)
                correct += (pred==yb).sum().item()
                total += yb.numel()
        acc = correct/total if total else 0.0
        if acc > best_val:
            best_val = acc
            torch.save({"state_dict": model.state_dict(),
                        "feat_cols": feat_cols,
                        "scaler_mean": scaler.mean_.tolist(),
                        "scaler_scale": scaler.scale_.tolist(),
                        "classes": ION_CLASSES}, os.path.join(args.outdir, "model.pt"))
        print(f"epoch {epoch:02d}  val_acc={acc:.3f}  best={best_val:.3f}")

    # (optional) temperature scaling on val set — quick fit
    # reload best model and fit model.log_temp to minimize NLL on val
    ckpt = torch.load(os.path.join(args.outdir, "model.pt"), map_location=device)
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
    torch.save({"state_dict": model.state_dict(),
                "feat_cols": feat_cols,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "classes": ION_CLASSES},
               os.path.join(args.outdir, "model.pt"))
    with open(os.path.join(args.outdir,"meta.json"),"w") as f:
        json.dump({"best_val_acc": best_val, "classes": ION_CLASSES}, f, indent=2)

if __name__ == "__main__":
    main()
