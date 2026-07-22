"""
train_mor_binary.py

Purpose:
    Train the baseline MOR / Non-MOR binary classification model using:
        TF-IDF Vectorizer + Logistic Regression

Input:
    data/processed/training/mor_binary_training_dataset.xlsx

Required columns:
    clean_brief_description
    mor_label
    group_based_split

Outputs:
    models/mor_binary_model.pkl
    models/mor_binary_vectorizer.pkl
    outputs/evaluation/mor_binary_evaluation.xlsx
    outputs/evaluation/mor_binary_confusion_matrix.png

Main metrics tracked:
    Accuracy
    Precision for MOR
    Recall for MOR
    F1-score for MOR
    Macro F1-score
    Confusion Matrix
    False Negatives
    False Positives

Run from project root:
    python src/train_mor_binary.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

def find_project_root() -> Path:
    """
    Find project root robustly.

    Normal expected location:
        MOR_Classification_Project/src/train_mor_binary.py

    If the script is run from a temporary folder during testing, it falls back
    to the current working directory.
    """
    script_path = Path(__file__).resolve()

    # If script is inside src/, project root is one level above src.
    if script_path.parent.name.lower() == "src":
        return script_path.parents[1]

    # Otherwise, search upward for data/processed/training.
    for parent in [script_path.parent] + list(script_path.parents):
        if (parent / "data" / "processed" / "training").exists():
            return parent

    # Fallback for standalone execution.
    return Path.cwd()


PROJECT_ROOT = find_project_root()

TRAINING_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "training" / "mor_binary_training_dataset.xlsx"
FALLBACK_TRAINING_DATA_FILE = Path("mor_binary_training_dataset.xlsx")

MODELS_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"

MODEL_OUTPUT_FILE = MODELS_DIR / "mor_binary_model.pkl"
VECTORIZER_OUTPUT_FILE = MODELS_DIR / "mor_binary_vectorizer.pkl"
EVALUATION_OUTPUT_FILE = EVALUATION_DIR / "mor_binary_evaluation.xlsx"
CONFUSION_MATRIX_PNG = EVALUATION_DIR / "mor_binary_confusion_matrix.png"


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

TEXT_COLUMN = "clean_brief_description"
TARGET_COLUMN = "mor_label"
SPLIT_COLUMN = "group_based_split"

MOR_CLASS = 1
NON_MOR_CLASS = 0

# Classification threshold for converting MOR probability into class label.
# Probability >= 0.45 means MOR, else Non-MOR.
MOR_DECISION_THRESHOLD = 0.45

# Review thresholds required by the project.
GENERAL_LOW_CONFIDENCE_THRESHOLD = 0.70
NON_MOR_LOW_CONFIDENCE_THRESHOLD = 0.80


# ============================================================
# 3. UTILITY FUNCTIONS
# ============================================================

def resolve_training_file() -> Path:
    """Return the available training data path."""
    if TRAINING_DATA_FILE.exists():
        return TRAINING_DATA_FILE
    if FALLBACK_TRAINING_DATA_FILE.exists():
        return FALLBACK_TRAINING_DATA_FILE
    raise FileNotFoundError(
        "Training data file not found. Checked:\n"
        f"1. {TRAINING_DATA_FILE}\n"
        f"2. {FALLBACK_TRAINING_DATA_FILE}\n"
    )


def ensure_output_dirs() -> None:
    """Create required output directories if they do not exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


def validate_columns(df: pd.DataFrame) -> None:
    """Validate that all required columns are available."""
    required_cols = [TEXT_COLUMN, TARGET_COLUMN, SPLIT_COLUMN]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in training dataset: {missing_cols}")


def load_training_data() -> pd.DataFrame:
    """Load and validate the binary MOR training dataset."""
    file_path = resolve_training_file()
    df = pd.read_excel(file_path, engine="openpyxl")
    validate_columns(df)

    # Basic cleaning and validation.
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).str.strip()
    df = df[df[TEXT_COLUMN] != ""].copy()

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df[df[TARGET_COLUMN].isin([0, 1])].copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    df[SPLIT_COLUMN] = df[SPLIT_COLUMN].fillna("").astype(str).str.lower().str.strip()
    df = df[df[SPLIT_COLUMN].isin(["train", "validation", "test"])].copy()

    if df.empty:
        raise ValueError("No usable rows found after cleaning training dataset.")

    return df


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset using group_based_split."""
    train_df = df[df[SPLIT_COLUMN] == "train"].copy()
    validation_df = df[df[SPLIT_COLUMN] == "validation"].copy()
    test_df = df[df[SPLIT_COLUMN] == "test"].copy()

    if train_df.empty:
        raise ValueError("Training split is empty. Check group_based_split values.")
    if validation_df.empty:
        raise ValueError("Validation split is empty. Check group_based_split values.")
    if test_df.empty:
        raise ValueError("Test split is empty. Check group_based_split values.")

    return train_df, validation_df, test_df


# ============================================================
# 4. TRAINING FUNCTIONS
# ============================================================

def build_vectorizer() -> TfidfVectorizer:
    """
    Build TF-IDF vectorizer.

    Settings used:
        ngram_range=(1, 2): captures single words and two-word aviation phrases.
        max_features=50000: keeps vocabulary manageable.
        min_df=2: ignores extremely rare words appearing in only one report.
        max_df=0.95: removes very common terms with little classification value.
    """
    return TfidfVectorizer(
        lowercase=False,  # text already cleaned/lowercased during preprocessing
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        norm="l2",
    )


def build_model() -> LogisticRegression:
    """
    Build Logistic Regression classifier.

    class_weight='balanced' is used because MOR and Non-MOR classes are not equally distributed.
    """
    return LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )


def train_model(train_df: pd.DataFrame) -> Tuple[TfidfVectorizer, LogisticRegression]:
    """Train TF-IDF vectorizer and Logistic Regression classifier."""
    vectorizer = build_vectorizer()
    model = build_model()

    X_train_text = train_df[TEXT_COLUMN].tolist()
    y_train = train_df[TARGET_COLUMN].values

    X_train_tfidf = vectorizer.fit_transform(X_train_text)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train_tfidf, y_train)

    return vectorizer, model


def predict_with_confidence(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
) -> pd.DataFrame:
    """Generate predictions, MOR probability and confidence scores."""
    output_df = df.copy()

    X = vectorizer.transform(output_df[TEXT_COLUMN].tolist())
    probabilities = model.predict_proba(X)

    # Model classes may be [0, 1]. Find class indices safely.
    class_to_index = {class_label: idx for idx, class_label in enumerate(model.classes_)}
    mor_idx = class_to_index[MOR_CLASS]
    non_mor_idx = class_to_index[NON_MOR_CLASS]

    output_df["mor_probability"] = probabilities[:, mor_idx]
    output_df["non_mor_probability"] = probabilities[:, non_mor_idx]

    output_df["predicted_mor_label"] = (
        output_df["mor_probability"] >= MOR_DECISION_THRESHOLD
    ).astype(int)

    output_df["predicted_label_text"] = np.where(
        output_df["predicted_mor_label"] == 1,
        "MOR",
        "Non-MOR",
    )

    output_df["actual_label_text"] = np.where(
        output_df[TARGET_COLUMN] == 1,
        "MOR",
        "Non-MOR",
    )

    # Confidence means probability of predicted class.
    output_df["prediction_confidence"] = np.where(
        output_df["predicted_mor_label"] == 1,
        output_df["mor_probability"],
        output_df["non_mor_probability"],
    )

    output_df["is_correct_prediction"] = (
        output_df[TARGET_COLUMN] == output_df["predicted_mor_label"]
    )

    output_df["error_type"] = "Correct"
    output_df.loc[
        (output_df[TARGET_COLUMN] == 1) & (output_df["predicted_mor_label"] == 0),
        "error_type",
    ] = "False Negative - Actual MOR predicted Non-MOR"
    output_df.loc[
        (output_df[TARGET_COLUMN] == 0) & (output_df["predicted_mor_label"] == 1),
        "error_type",
    ] = "False Positive - Actual Non-MOR predicted MOR"

    output_df = add_review_flags(output_df)

    return output_df


def add_review_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add human review flags based on required project logic.

    Rules implemented:
        1. If prediction confidence < 0.70 -> Human Review Required
        2. If predicted Non-MOR but confidence < 0.80 -> Human Review Recommended
        3. If rule engine says MOR but ML says Non-MOR -> Mandatory Review

    Note:
        Rule-engine output is optional at this binary-training stage.
        If a column called rule_based_mor exists, it will be used.
        Expected values for rule_based_mor: 1 / 0 / True / False
    """
    output_df = df.copy()

    output_df["review_required"] = "No"
    output_df["review_priority"] = "None"
    output_df["review_reason"] = "No review rule triggered"

    # Rule 1: General low-confidence prediction.
    mask_low_confidence = output_df["prediction_confidence"] < GENERAL_LOW_CONFIDENCE_THRESHOLD
    output_df.loc[mask_low_confidence, "review_required"] = "Yes"
    output_df.loc[mask_low_confidence, "review_priority"] = "Required"
    output_df.loc[mask_low_confidence, "review_reason"] = "Prediction confidence below 0.70"

    # Rule 2: Predicted Non-MOR with confidence below 0.80.
    mask_non_mor_low_conf = (
        (output_df["predicted_mor_label"] == 0)
        & (output_df["prediction_confidence"] < NON_MOR_LOW_CONFIDENCE_THRESHOLD)
    )
    output_df.loc[mask_non_mor_low_conf, "review_required"] = "Yes"
    output_df.loc[mask_non_mor_low_conf, "review_priority"] = "Recommended"
    output_df.loc[mask_non_mor_low_conf, "review_reason"] = (
        "Predicted Non-MOR with confidence below 0.80"
    )

    # Rule 3: Rule engine conflict, if available.
    if "rule_based_mor" in output_df.columns:
        rule_based_values = output_df["rule_based_mor"].astype(str).str.lower().str.strip()
        mask_rule_says_mor = rule_based_values.isin(["1", "true", "yes", "mor"])
        mask_rule_conflict = mask_rule_says_mor & (output_df["predicted_mor_label"] == 0)

        output_df.loc[mask_rule_conflict, "review_required"] = "Yes"
        output_df.loc[mask_rule_conflict, "review_priority"] = "Mandatory"
        output_df.loc[mask_rule_conflict, "review_reason"] = (
            "Rule engine indicates MOR but ML predicted Non-MOR"
        )

    return output_df


# ============================================================
# 5. EVALUATION FUNCTIONS
# ============================================================

def compute_metrics(prediction_df: pd.DataFrame, split_name: str) -> Dict[str, float]:
    """Compute core binary-classification metrics."""
    y_true = prediction_df[TARGET_COLUMN].values
    y_pred = prediction_df["predicted_mor_label"].values

    metrics = {
        "split": split_name,
        "rows": len(prediction_df),
        "actual_mor_rows": int((y_true == 1).sum()),
        "actual_non_mor_rows": int((y_true == 0).sum()),
        "predicted_mor_rows": int((y_pred == 1).sum()),
        "predicted_non_mor_rows": int((y_pred == 0).sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_for_mor": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_for_mor": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_score_for_mor": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "macro_f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics.update({
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    })

    return metrics


def classification_report_dataframe(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """Return sklearn classification report as DataFrame."""
    y_true = prediction_df[TARGET_COLUMN].values
    y_pred = prediction_df["predicted_mor_label"].values

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Non-MOR", "MOR"],
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "class_or_average"})


def confusion_matrix_dataframe(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """Return confusion matrix as DataFrame."""
    y_true = prediction_df[TARGET_COLUMN].values
    y_pred = prediction_df["predicted_mor_label"].values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["Actual Non-MOR", "Actual MOR"],
        columns=["Predicted Non-MOR", "Predicted MOR"],
    )


def plot_confusion_matrix(prediction_df: pd.DataFrame, output_path: Path) -> None:
    """Save confusion matrix PNG for the test split."""
    cm_df = confusion_matrix_dataframe(prediction_df)
    cm = cm_df.values

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)

    ax.set_title("MOR Binary Classifier - Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-MOR", "MOR"])
    ax.set_yticklabels(["Non-MOR", "MOR"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_review_summary(prediction_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Summarize review flags for a split."""
    summary = (
        prediction_df.groupby(["review_required", "review_priority", "review_reason"])
        .size()
        .reset_index(name="row_count")
    )
    summary.insert(0, "split", split_name)
    return summary


def create_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create dataset-level summary before training."""
    summary_rows = []

    for split in ["train", "validation", "test"]:
        split_df = df[df[SPLIT_COLUMN] == split]
        summary_rows.append({
            "split": split,
            "rows": len(split_df),
            "mor_rows": int((split_df[TARGET_COLUMN] == 1).sum()),
            "non_mor_rows": int((split_df[TARGET_COLUMN] == 0).sum()),
            "mor_percentage": round(float((split_df[TARGET_COLUMN] == 1).mean() * 100), 2) if len(split_df) else 0,
            "avg_text_words": round(float(split_df[TEXT_COLUMN].str.split().str.len().mean()), 2) if len(split_df) else 0,
        })

    return pd.DataFrame(summary_rows)


# ============================================================
# 6. SAVE FUNCTIONS
# ============================================================

def save_models(vectorizer: TfidfVectorizer, model: LogisticRegression) -> None:
    """Save trained vectorizer and model."""
    joblib.dump(model, MODEL_OUTPUT_FILE)
    joblib.dump(vectorizer, VECTORIZER_OUTPUT_FILE)


def save_evaluation_workbook(
    original_df: pd.DataFrame,
    train_predictions: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
) -> None:
    """Save all evaluation outputs into one Excel workbook."""

    metrics_df = pd.DataFrame([
        compute_metrics(train_predictions, "train"),
        compute_metrics(validation_predictions, "validation"),
        compute_metrics(test_predictions, "test"),
    ])

    validation_report_df = classification_report_dataframe(validation_predictions)
    test_report_df = classification_report_dataframe(test_predictions)

    validation_cm_df = confusion_matrix_dataframe(validation_predictions).reset_index().rename(columns={"index": "Actual/Predicted"})
    test_cm_df = confusion_matrix_dataframe(test_predictions).reset_index().rename(columns={"index": "Actual/Predicted"})

    false_negatives_df = test_predictions[
        (test_predictions[TARGET_COLUMN] == 1)
        & (test_predictions["predicted_mor_label"] == 0)
    ].copy()

    false_positives_df = test_predictions[
        (test_predictions[TARGET_COLUMN] == 0)
        & (test_predictions["predicted_mor_label"] == 1)
    ].copy()

    review_summary_df = pd.concat(
        [
            create_review_summary(train_predictions, "train"),
            create_review_summary(validation_predictions, "validation"),
            create_review_summary(test_predictions, "test"),
        ],
        ignore_index=True,
    )

    dataset_summary_df = create_dataset_summary(original_df)

    model_config_df = pd.DataFrame([
        {"parameter": "algorithm", "value": "TF-IDF + Logistic Regression"},
        {"parameter": "text_column", "value": TEXT_COLUMN},
        {"parameter": "target_column", "value": TARGET_COLUMN},
        {"parameter": "split_column", "value": SPLIT_COLUMN},
        {"parameter": "mor_decision_threshold", "value": MOR_DECISION_THRESHOLD},
        {"parameter": "general_low_confidence_threshold", "value": GENERAL_LOW_CONFIDENCE_THRESHOLD},
        {"parameter": "non_mor_low_confidence_threshold", "value": NON_MOR_LOW_CONFIDENCE_THRESHOLD},
        {"parameter": "tfidf_ngram_range", "value": "(1, 2)"},
        {"parameter": "tfidf_max_features", "value": 50000},
        {"parameter": "tfidf_min_df", "value": 2},
        {"parameter": "tfidf_max_df", "value": 0.95},
        {"parameter": "logistic_regression_class_weight", "value": "balanced"},
        {"parameter": "logistic_regression_solver", "value": "liblinear"},
        {"parameter": "model_classes", "value": json.dumps(model.classes_.tolist())},
        {"parameter": "vocabulary_size", "value": len(vectorizer.vocabulary_)},
    ])

    # Keep prediction sheets compact but useful.
    prediction_cols = [
        "occurrence_internal_id",
        "REPORT NO.",
        TEXT_COLUMN,
        "brief_description",
        TARGET_COLUMN,
        "actual_label_text",
        "predicted_mor_label",
        "predicted_label_text",
        "mor_probability",
        "non_mor_probability",
        "prediction_confidence",
        "is_correct_prediction",
        "error_type",
        "review_required",
        "review_priority",
        "review_reason",
        "group_based_split",
        "time_based_split",
        "duplicate_group_id",
        "is_duplicate_summary",
        "REPORT TYPE",
        "date_of_occurrence_parsed",
        "MONTH",
        "AIRCRAFT TYPE",
        "SECTOR",
        "AIRPORT",
        "PHASE OF FLIGHT",
    ]
    prediction_cols = [col for col in prediction_cols if col in test_predictions.columns]

    with pd.ExcelWriter(EVALUATION_OUTPUT_FILE, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="Metrics_Summary", index=False)
        dataset_summary_df.to_excel(writer, sheet_name="Dataset_Summary", index=False)
        validation_report_df.to_excel(writer, sheet_name="Validation_Report", index=False)
        test_report_df.to_excel(writer, sheet_name="Test_Report", index=False)
        validation_cm_df.to_excel(writer, sheet_name="Validation_CM", index=False)
        test_cm_df.to_excel(writer, sheet_name="Test_CM", index=False)
        false_negatives_df[prediction_cols].to_excel(writer, sheet_name="False_Negatives_Test", index=False)
        false_positives_df[prediction_cols].to_excel(writer, sheet_name="False_Positives_Test", index=False)
        review_summary_df.to_excel(writer, sheet_name="Review_Summary", index=False)
        validation_predictions[prediction_cols].to_excel(writer, sheet_name="Validation_Predictions", index=False)
        test_predictions[prediction_cols].to_excel(writer, sheet_name="Test_Predictions", index=False)
        model_config_df.to_excel(writer, sheet_name="Model_Config", index=False)


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

def main() -> None:
    ensure_output_dirs()

    print("Loading MOR binary training dataset...")
    df = load_training_data()

    print("Splitting dataset using group_based_split...")
    train_df, validation_df, test_df = split_dataset(df)

    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Test rows: {len(test_df)}")

    print("Training TF-IDF + Logistic Regression model...")
    vectorizer, model = train_model(train_df)

    print("Generating predictions...")
    train_predictions = predict_with_confidence(train_df, vectorizer, model)
    validation_predictions = predict_with_confidence(validation_df, vectorizer, model)
    test_predictions = predict_with_confidence(test_df, vectorizer, model)

    print("Saving trained model and vectorizer...")
    save_models(vectorizer, model)

    print("Saving evaluation workbook...")
    save_evaluation_workbook(
        original_df=df,
        train_predictions=train_predictions,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        vectorizer=vectorizer,
        model=model,
    )

    print("Saving confusion matrix image...")
    plot_confusion_matrix(test_predictions, CONFUSION_MATRIX_PNG)

    test_metrics = compute_metrics(test_predictions, "test")

    print("\nTraining completed successfully.")
    print(f"Model saved to: {MODEL_OUTPUT_FILE}")
    print(f"Vectorizer saved to: {VECTORIZER_OUTPUT_FILE}")
    print(f"Evaluation workbook saved to: {EVALUATION_OUTPUT_FILE}")
    print(f"Confusion matrix image saved to: {CONFUSION_MATRIX_PNG}")

    print("\nTest-set key metrics:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision for MOR: {test_metrics['precision_for_mor']:.4f}")
    print(f"Recall for MOR: {test_metrics['recall_for_mor']:.4f}")
    print(f"F1-score for MOR: {test_metrics['f1_score_for_mor']:.4f}")
    print(f"Macro F1-score: {test_metrics['macro_f1_score']:.4f}")
    print(f"False Negatives: {test_metrics['false_negatives']}")
    print(f"False Positives: {test_metrics['false_positives']}")


if __name__ == "__main__":
    main()
