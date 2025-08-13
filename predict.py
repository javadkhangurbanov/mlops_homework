import argparse
import os

import joblib
import pandas as pd


def load_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {ext}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    pipe = joblib.load(args.model)
    df = load_any(args.data)
    proba = pipe.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = df.copy()
    out["pred_proba"] = proba
    out["pred_label"] = pred
    out.to_csv(args.out, index=False)
    print(f"Wrote predictions → {args.out}")


if __name__ == "__main__":
    main()
