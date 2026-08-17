import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_FRACTION = 0.2

LEAKY_FEATURES = ["duration"]

TARGET_COLUMN = "y"
POSITIVE_LABEL = "yes"

MODEL_DIR = Path(__file__).resolve().parent
BASE_DIR = MODEL_DIR.parent
DATA_FILE = BASE_DIR / "data" / "bank-full.csv"
TEST_CSV = BASE_DIR / "test_data.csv"

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}."
        )
    return pd.read_csv(DATA_FILE, sep=";")


def get_features_and_target():
    frame = load_dataset().drop(columns=LEAKY_FEATURES)
    target = (frame.pop(TARGET_COLUMN) == POSITIVE_LABEL).astype(int)
    return frame, target


def get_train_test():
    features, target = get_features_and_target()
    return train_test_split(
        features, target,
        test_size=TEST_FRACTION,
        stratify=target,
        random_state=RANDOM_STATE,
    )


def feature_groups(features):
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    return numeric, categorical


def build_preprocessor(features):
    numeric, categorical = feature_groups(features)
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                              sparse_output=False), categorical),
    ])


def build_pipeline(classifier, features):
    return Pipeline([("preprocess", build_preprocessor(features)),
                     ("classifier", classifier)])


def evaluate(pipeline, X, y_true):
    y_pred = pipeline.predict(X)
    y_score = pipeline.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def report(model_name, metrics):
    print(f"\n{model_name}")
    print("-" * len(model_name))
    for metric in METRIC_NAMES:
        print(f"  {metric:<10} {metrics[metric]:.4f}")


def save_pipeline(pipeline, file_stem):
    path = MODEL_DIR / f"{file_stem}.pkl"
    joblib.dump(pipeline, path, compress=3)
    print(f"  saved -> {path.name} ({path.stat().st_size / 1e6:.2f} MB)")
    return path


def export_test_csv(X_test, y_test):
    export = X_test.copy()
    export[TARGET_COLUMN] = np.where(y_test == 1, POSITIVE_LABEL, "no")
    export.to_csv(TEST_CSV, index=False)
    print(f"  saved -> {TEST_CSV.name} ({len(export):,} rows)")


def write_schema(features, file_stems):
    numeric, categorical = feature_groups(features)
    (MODEL_DIR / "schema.json").write_text(json.dumps({
        "target": TARGET_COLUMN,
        "positive_label": POSITIVE_LABEL,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "all_features": features.columns.tolist(),
        "files": file_stems,
    }, indent=2))


def train_and_save(model_name, classifier, file_stem):
    X_train, X_test, y_train, y_test = get_train_test()
    pipeline = build_pipeline(classifier, X_train)
    pipeline.fit(X_train, y_train)

    metrics = evaluate(pipeline, X_test, y_test)
    report(model_name, metrics)
    save_pipeline(pipeline, file_stem)
    return pipeline, metrics
