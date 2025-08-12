import argparse, joblib, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to saved joblib model")
    ap.add_argument("--data",  required=True, help="CSV without target (or target ignored)")
    ap.add_argument("--out",   default="predictions.csv")
    args = ap.parse_args()

    pipe = joblib.load(args.model)
    df = pd.read_csv(args.data)
    proba = pipe.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = df.copy()
    out["pred_proba"] = proba
    out["pred_label"] = pred
    out.to_csv(args.out, index=False)
    print(f"Wrote predictions → {args.out}")

if __name__ == "__main__":
    main()
