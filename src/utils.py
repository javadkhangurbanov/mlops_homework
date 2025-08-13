import json
import os

import pandas as pd


def load_xy(file_path: str, target_col: str = "target"):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    return X, y


def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
