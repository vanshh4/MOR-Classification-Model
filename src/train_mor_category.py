"""
train_mor_category.py

Train MOR category/sub-category classifiers using TF-IDF + Logistic Regression.

Input:
    data/processed/training/mor_category_training_dataset.xlsx

Outputs:
    models/mor_category_level1_model.pkl
    models/mor_category_level1_vectorizer.pkl
    models/mor_event_type_model.pkl
    models/mor_event_type_vectorizer.pkl
    models/mor_report_title_model.pkl
    models/mor_report_title_vectorizer.pkl
    outputs/evaluation/mor_category_evaluation.xlsx

Run:
    python src/train_mor_category.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    if script_path.parent.name.lower() == "src":
        return script_path.parents[1]
    for parent in [script_path.parent] + list(script_path.parents):
        if (parent / "data" / "processed" / "training").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "training" / "mor_category_training_dataset.xlsx"
FALLBACK_DATA_FILE = Path("mor_category_training_dataset.xlsx")
MODELS_DIR = PROJECT_ROOT / "models"
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"
EVAL_FILE = EVAL_DIR / "mor_category_evaluation.xlsx"

TEXT_COL = "clean_brief_description"
SPLIT_COL = "group_based_split"
UNKNOWN_VALUES = {"", "UNKNOWN", "NAN", "NA", "N/A", "NONE", "-"}

TARGETS = [
    {
        "model_key": "mor_category_level1",
        "target_col": "iata_level_1_normalized",
        "description": "Broad MOR category / IATA IDX Parent Level 1",
        "min_train_examples_per_class": 2,
    },
    {
        "model_key": "mor_event_type",
        "target_col": "iata_event_level_3_normalized",
        "description": "MOR event type / IATA IDX Event Type Level 3",
        "min_train_examples_per_class": 2,
    },
    {
        "model_key": "mor_report_title",
        "target_col": "report_title_normalized",
        "description": "MOR report title / business-facing sub-category",
        "min_train_examples_per_class": 2,
    },
]


def resolve_data_file() -> Path:
    if DATA_FILE.exists():
        return DATA_FILE

    if FALLBACK_DATA_FILE.exists():
        return FALLBACK_DATA_FILE

    raise FileNotFoundError(
        "Dataset not found. Checked:\n"
        f"1. {DATA_FILE}\n"
        f"2. {FALLBACK_DATA_FILE}"
    )


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


def clean_label(value: object) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).replace(" ", " ").strip()
    text = " ".join(text.split())
    if text.upper() in UNKNOWN_VALUES:
        return "UNKNOWN"
    return text


def load_data() -> pd.DataFrame:
    path = resolve_data_file()
    df = pd.read_excel(path, engine="openpyxl")
    required = [TEXT_COL, SPLIT_COL] + [t["target_col"] for t in TARGETS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).str.strip()
    df = df[df[TEXT_COL] != ""].copy()
    df[SPLIT_COL] = df[SPLIT_COL].fillna("").astype(str).str.lower().str.strip()
    df = df[df[SPLIT_COL].isin(["train", "validation", "test"])].copy()
    return df


def filter_for_target(df: pd.DataFrame, target_col: str, min_count: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work[target_col] = work[target_col].apply(clean_label)

    excluded = []

    unknown_mask = work[target_col].str.upper().isin(UNKNOWN_VALUES)
    if unknown_mask.any():
        tmp = work[unknown_mask].copy()
        tmp["exclusion_reason"] = f"UNKNOWN label in {target_col}"
        excluded.append(tmp)
    work = work[~unknown_mask].copy()

    train_counts = work[work[SPLIT_COL] == "train"][target_col].value_counts()
    valid_classes = train_counts[train_counts >= min_count].index.tolist()

    invalid_mask = ~work[target_col].isin(valid_classes)
    if invalid_mask.any():
        tmp = work[invalid_mask].copy()
        tmp["exclusion_reason"] = f"Label has fewer than {min_count} train examples or unseen in train"
        excluded.append(tmp)
    work = work[~invalid_mask].copy()

    if work[work[SPLIT_COL] == "train"][target_col].nunique() < 2:
        raise ValueError(f"Cannot train {target_col}: fewer than 2 train classes after filtering.")

    excluded_df = pd.concat(excluded, ignore_index=True) if excluded else pd.DataFrame()
    return work, excluded_df


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        norm="l2",
    )


def build_model() -> LogisticRegression:
    return LogisticRegression(
        max_iter=10000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )


def train_model(train_df: pd.DataFrame, target_col: str):
    vectorizer = build_vectorizer()
    model = build_model()
    x_train = vectorizer.fit_transform(train_df[TEXT_COL].tolist())
    y_train = train_df[target_col].astype(str).values
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(x_train, y_train)
    return vectorizer, model


def predict(df: pd.DataFrame, target_col: str, model_key: str, vectorizer, model) -> pd.DataFrame:
    out = df.copy()
    x = vectorizer.transform(out[TEXT_COL].tolist())
    pred = model.predict(x)
    proba = model.predict_proba(x).max(axis=1)
    out[f"{model_key}_actual"] = out[target_col].astype(str)
    out[f"{model_key}_predicted"] = pred
    out[f"{model_key}_confidence"] = proba
    out[f"{model_key}_is_correct"] = out[f"{model_key}_actual"] == out[f"{model_key}_predicted"]
    return out


def metrics(pred_df: pd.DataFrame, model_key: str, split: str) -> Dict:
    y_true = pred_df[f"{model_key}_actual"].astype(str)
    y_pred = pred_df[f"{model_key}_predicted"].astype(str)
    return {
        "model_key": model_key,
        "split": split,
        "rows": len(pred_df),
        "unique_actual_classes": y_true.nunique(),
        "unique_predicted_classes": y_pred.nunique(),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "avg_confidence": pred_df[f"{model_key}_confidence"].mean(),
    }


def report_df(pred_df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    y_true = pred_df[f"{model_key}_actual"].astype(str)
    y_pred = pred_df[f"{model_key}_predicted"].astype(str)
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    out = pd.DataFrame(rep).transpose().reset_index().rename(columns={"index": "class_or_average"})
    out.insert(0, "model_key", model_key)
    return out


def top_confusions(pred_df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    actual = f"{model_key}_actual"
    predicted = f"{model_key}_predicted"
    err = pred_df[pred_df[actual] != pred_df[predicted]].copy()
    if err.empty:
        return pd.DataFrame(columns=["model_key", "actual", "predicted", "count"])
    out = (
        err.groupby([actual, predicted])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(50)
        .rename(columns={actual: "actual", predicted: "predicted"})
    )
    out.insert(0, "model_key", model_key)
    return out


def class_distribution(df: pd.DataFrame, target_col: str, model_key: str) -> pd.DataFrame:
    out = df.groupby([SPLIT_COL, target_col]).size().reset_index(name="row_count")
    out = out.rename(columns={target_col: "label"})
    out.insert(0, "model_key", model_key)
    return out


def prediction_cols(df: pd.DataFrame, model_key: str) -> List[str]:
    cols = [
        "occurrence_internal_id", "REPORT NO.", TEXT_COL, "brief_description",
        SPLIT_COL, "time_based_split", "duplicate_group_id", "REPORT TYPE",
        "date_of_occurrence_parsed", "MONTH", "AIRCRAFT TYPE", "SECTOR", "AIRPORT", "PHASE OF FLIGHT",
        f"{model_key}_actual", f"{model_key}_predicted", f"{model_key}_confidence", f"{model_key}_is_correct",
    ]
    return [c for c in cols if c in df.columns]


def main() -> None:
    ensure_dirs()
    df = load_data()

    all_metrics = []
    all_reports = []
    all_confusions = []
    all_distributions = []
    all_configs = []
    all_excluded = []
    prediction_sheets = {}

    for target in TARGETS:
        model_key = target["model_key"]
        target_col = target["target_col"]
        min_count = target["min_train_examples_per_class"]

        print(f"\nTraining {model_key} using target: {target_col}")

        prepared, excluded = filter_for_target(df, target_col, min_count)

        if not excluded.empty:
            excluded.insert(0, "model_key", model_key)
            all_excluded.append(excluded)

        train_df = prepared[prepared[SPLIT_COL] == "train"].copy()
        val_df = prepared[prepared[SPLIT_COL] == "validation"].copy()
        test_df = prepared[prepared[SPLIT_COL] == "test"].copy()

        vectorizer, model = train_model(train_df, target_col)

        joblib.dump(model, MODELS_DIR / f"{model_key}_model.pkl")
        joblib.dump(vectorizer, MODELS_DIR / f"{model_key}_vectorizer.pkl")

        train_pred = predict(train_df, target_col, model_key, vectorizer, model)
        val_pred = predict(val_df, target_col, model_key, vectorizer, model)
        test_pred = predict(test_df, target_col, model_key, vectorizer, model)

        all_metrics.extend([
            metrics(train_pred, model_key, "train"),
            metrics(val_pred, model_key, "validation"),
            metrics(test_pred, model_key, "test"),
        ])

        all_reports.append(report_df(test_pred, model_key))
        all_confusions.append(top_confusions(test_pred, model_key))
        all_distributions.append(class_distribution(prepared, target_col, model_key))

        prediction_sheets[f"{model_key}_test"] = test_pred[
            prediction_cols(test_pred, model_key)
        ]

        prediction_sheets[f"{model_key}_val"] = val_pred[
            prediction_cols(val_pred, model_key)
        ]

        all_configs.append({
            "model_key": model_key,
            "description": target["description"],
            "target_column": target_col,
            "training_rows_after_filtering": len(prepared),
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
            "unique_train_classes": train_df[target_col].nunique(),
            "model_file": str(MODELS_DIR / f"{model_key}_model.pkl"),
            "vectorizer_file": str(MODELS_DIR / f"{model_key}_vectorizer.pkl"),
            "algorithm": "TF-IDF + Logistic Regression",
            "model_classes": json.dumps([str(x) for x in model.classes_]),
        })

    metrics_df = pd.DataFrame(all_metrics)

    reports = (
        pd.concat(all_reports, ignore_index=True)
        if all_reports
        else pd.DataFrame()
    )

    confusions = (
        pd.concat(all_confusions, ignore_index=True)
        if all_confusions
        else pd.DataFrame()
    )

    distributions = (
        pd.concat(all_distributions, ignore_index=True)
        if all_distributions
        else pd.DataFrame()
    )

    configs = pd.DataFrame(all_configs)

    excluded_df = (
        pd.concat(all_excluded, ignore_index=True)
        if all_excluded
        else pd.DataFrame()
    )

    with pd.ExcelWriter(EVAL_FILE, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="Metrics_Summary", index=False)
        configs.to_excel(writer, sheet_name="Model_Config", index=False)
        distributions.to_excel(writer, sheet_name="Class_Distribution", index=False)
        reports.to_excel(writer, sheet_name="Test_Class_Report", index=False)
        confusions.to_excel(writer, sheet_name="Top_Confusions_Test", index=False)

        if not excluded_df.empty:
            cols = [
                "model_key",
                "occurrence_internal_id",
                "REPORT NO.",
                TEXT_COL,
                "brief_description",
                SPLIT_COL,
                "exclusion_reason",
            ]

            cols = [c for c in cols if c in excluded_df.columns]

            excluded_df[cols].to_excel(
                writer,
                sheet_name="Excluded_Rows",
                index=False,
            )

        for name, sheet in prediction_sheets.items():
            sheet.to_excel(
                writer,
                sheet_name=name[:31],
                index=False,
            )

    print("\nMOR category training completed successfully.")
    print(f"Evaluation saved to: {EVAL_FILE}")
    print(metrics_df[metrics_df["split"] == "test"].to_string(index=False))

if __name__ == "__main__":
    main()
