#!/usr/bin/env python3
import argparse, json
import numpy as np, pandas as pd, torch
from model import IonWaterMLP

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ionwater_env_model/model.pt")
    ap.add_argument("--features-csv", required=True, help="features for candidates (same columns used in training)")
    ap.add_argument("--out-csv", default="preds_env.csv")
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu")
    feat_cols = ck["feat_cols"]; classes = ck["classes"]
    mean = np.array(ck["scaler_mean"]); scale = np.array(ck["scaler_scale"])
    df = pd.read_csv(args.features_csv)
    X = df[feat_cols].to_numpy(dtype=np.float32)
    X = (X - mean) / scale

    model = IonWaterMLP(in_dim=X.shape[1], n_classes=len(classes))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32)
        logits, aux = model(xb, calibrate=True)   # use calibrated temperature
        probs = torch.softmax(logits, dim=-1).numpy()
        aux_p = torch.softmax(aux, dim=-1).numpy()  # [:,1] = ion

    top_idx = probs.argmax(axis=1)
    df["pred_class"] = [classes[i] for i in top_idx]
    df["confidence"] = probs.max(axis=1)
    # optional diagnostics
    for i, cls in enumerate(classes):
        df[f"p_{cls}"] = probs[:, i]
    df["p_ion"] = aux_p[:,1]

    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} with {len(df)} rows")

if __name__ == "__main__":
    main()
