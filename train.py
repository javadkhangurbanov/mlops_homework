import argparse, os, joblib, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from src.features import make_preprocessor
from src.model import make_model
from src.utils import load_xy, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with 'target' column")
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X, y = load_xy(args.data)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pos = int(y_tr.sum()); neg = int(len(y_tr) - pos)
    spw = (neg / pos) if pos else 1.0

    pre = make_preprocessor()
    xgb = make_model(scale_pos_weight=spw)
    pipe = Pipeline([("preprocessor", pre), ("model", xgb)])

    param_distributions = {
        "model__n_estimators":      np.arange(200, 900, 100),
        "model__learning_rate":     np.linspace(0.01, 0.2, 10),
        "model__max_depth":         [3, 4, 5, 6, 7],
        "model__min_child_weight":  [1, 2, 3, 5],
        "model__subsample":         np.linspace(0.6, 1.0, 5),
        "model__colsample_bytree":  np.linspace(0.6, 1.0, 5),
        "model__gamma":             [0, 1, 2],
        "model__reg_alpha":         [0, 1e-3, 1e-2, 1e-1, 1],
        "model__reg_lambda":        [0.5, 1, 2, 5, 10],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=args.iters,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
        random_state=42,
    )
    search.fit(X_tr, y_tr)

    best_pipe = search.best_estimator_
    proba = best_pipe.predict_proba(X_te)[:, 1]
    metrics = {
        "cv_roc_auc": float(search.best_score_),
        "holdout_roc_auc": float(roc_auc_score(y_te, proba)),
        "holdout_pr_auc": float(average_precision_score(y_te, proba)),
        "best_params": {k: (float(v) if hasattr(v, "__float__") else v) for k, v in search.best_params_.items()},
    }

    model_path = os.path.join(args.outdir, "model.joblib")
    metrics_path = os.path.join(args.outdir, "metrics.json")
    joblib.dump(best_pipe, model_path)
    save_json(metrics_path, metrics)
    print(f"Saved model → {model_path}")
    print(f"Saved metrics → {metrics_path}")

if __name__ == "__main__":
    main()
