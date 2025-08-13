from xgboost import XGBClassifier


def make_model(scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        enable_categorical=False,
        scale_pos_weight=scale_pos_weight,
    )
