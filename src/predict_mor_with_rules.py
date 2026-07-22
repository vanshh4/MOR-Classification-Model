"""
predict_mor_with_rules.py

Purpose:
    Unified inference script that combines:
        1. Trained MOR Binary ML Model
        2. MOR Rule Engine
        3. MOR Category/Sub-category Models

Core rule:
    If rule_engine says MOR and ML model says Non-MOR:
        Mandatory Review

Inputs supported:
    1. Single report text from terminal
    2. Excel file containing report text column

Expected project files:
    models/mor_binary_model.pkl
    models/mor_binary_vectorizer.pkl

    models/mor_category_level1_model.pkl
    models/mor_category_level1_vectorizer.pkl

    models/mor_event_type_model.pkl
    models/mor_event_type_vectorizer.pkl

    models/mor_report_title_model.pkl
    models/mor_report_title_vectorizer.pkl

    src/rule_engine.py

Default Excel input:
    data/processed/volrep/combined_volrep_2026H1_refined.xlsx

Default Excel output:
    outputs/predictions/volrep_mor_predictions_with_rules.xlsx

Run examples:
    python src/predict_mor_with_rules.py --text "Aircraft experienced bird strike during climb."

    python src/predict_mor_with_rules.py ^
        --input data/processed/volrep/combined_volrep_2026H1_refined.xlsx ^
        --text-column clean_report_text ^
        --output outputs/predictions/volrep_mor_predictions_with_rules.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd


# ============================================================
# 1. IMPORT RULE ENGINE
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from rule_engine import RuleEngine, combine_ml_and_rule_decision  # noqa: E402


# ============================================================
# 2. PROJECT PATH CONFIGURATION
# ============================================================

def find_project_root() -> Path:
    """
    Find project root based on expected src/ location.

    Expected:
        MOR_Classification_Project/src/predict_mor_with_rules.py
    """
    script_path = Path(__file__).resolve()

    if script_path.parent.name.lower() == "src":
        return script_path.parents[1]

    for parent in [script_path.parent] + list(script_path.parents):
        if (parent / "models").exists() and (parent / "src").exists():
            return parent

    return Path.cwd()


PROJECT_ROOT = find_project_root()


# ============================================================
# 3. MODEL PATHS
# ============================================================

# Binary MOR model
BINARY_MODEL_FILE = PROJECT_ROOT / "models" / "mor_binary_model.pkl"
BINARY_VECTORIZER_FILE = PROJECT_ROOT / "models" / "mor_binary_vectorizer.pkl"

# Category/sub-category models
CATEGORY_MODEL_FILES = {
    "category_level1": {
        "model": PROJECT_ROOT / "models" / "mor_category_level1_model.pkl",
        "vectorizer": PROJECT_ROOT / "models" / "mor_category_level1_vectorizer.pkl",
    },
    "event_type": {
        "model": PROJECT_ROOT / "models" / "mor_event_type_model.pkl",
        "vectorizer": PROJECT_ROOT / "models" / "mor_event_type_vectorizer.pkl",
    },
    "report_title": {
        "model": PROJECT_ROOT / "models" / "mor_report_title_model.pkl",
        "vectorizer": PROJECT_ROOT / "models" / "mor_report_title_vectorizer.pkl",
    },
}

# Default data paths
DEFAULT_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "volrep"
    / "combined_volrep_2026H1_refined.xlsx"
)

DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "volrep_mor_predictions_with_rules.xlsx"
)


# ============================================================
# 4. CONFIGURATION
# ============================================================

DEFAULT_TEXT_COLUMN = "clean_report_text"

MOR_CLASS = 1
NON_MOR_CLASS = 0

MOR_DECISION_THRESHOLD = 0.50

GENERAL_LOW_CONFIDENCE_THRESHOLD = 0.70
NON_MOR_LOW_CONFIDENCE_THRESHOLD = 0.80


# ============================================================
# 5. LOAD MODEL ARTIFACTS
# ============================================================

def load_binary_model_artifacts():
    """
    Load trained MOR binary model and TF-IDF vectorizer.
    """
    if not BINARY_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Binary model file not found: {BINARY_MODEL_FILE}\n"
            "Run python src/train_mor_binary.py first."
        )

    if not BINARY_VECTORIZER_FILE.exists():
        raise FileNotFoundError(
            f"Binary vectorizer file not found: {BINARY_VECTORIZER_FILE}\n"
            "Run python src/train_mor_binary.py first."
        )

    model = joblib.load(BINARY_MODEL_FILE)
    vectorizer = joblib.load(BINARY_VECTORIZER_FILE)

    return model, vectorizer


def load_category_model_artifacts() -> dict:
    """
    Load trained MOR category/sub-category models.

    Returns:
        Dictionary containing model and vectorizer for each category target.

    If a model is missing, the script skips that category model
    instead of stopping the entire pipeline.
    """
    category_artifacts = {}

    for key, paths in CATEGORY_MODEL_FILES.items():
        model_path = paths["model"]
        vectorizer_path = paths["vectorizer"]

        if model_path.exists() and vectorizer_path.exists():
            category_artifacts[key] = {
                "model": joblib.load(model_path),
                "vectorizer": joblib.load(vectorizer_path),
            }
        else:
            print(
                f"Warning: Category model artifacts not found for '{key}'. "
                f"Expected:\n"
                f"  Model: {model_path}\n"
                f"  Vectorizer: {vectorizer_path}\n"
                f"Skipping this category prediction."
            )

    return category_artifacts


def ensure_output_dir(output_path: Path) -> None:
    """
    Create output directory if it does not already exist.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 6. BINARY MOR ML PREDICTION
# ============================================================

def predict_ml_for_text(report_text: str, model, vectorizer) -> Dict:
    """
    Run trained binary MOR ML model on one report text.
    """
    text = "" if report_text is None else str(report_text).strip()

    X = vectorizer.transform([text])
    probabilities = model.predict_proba(X)[0]

    class_to_index = {
        class_label: idx
        for idx, class_label in enumerate(model.classes_)
    }

    mor_idx = class_to_index[MOR_CLASS]
    non_mor_idx = class_to_index[NON_MOR_CLASS]

    mor_probability = float(probabilities[mor_idx])
    non_mor_probability = float(probabilities[non_mor_idx])

    predicted_mor_label = int(mor_probability >= MOR_DECISION_THRESHOLD)

    predicted_label_text = (
        "MOR"
        if predicted_mor_label == 1
        else "Non-MOR"
    )

    prediction_confidence = (
        mor_probability
        if predicted_mor_label == 1
        else non_mor_probability
    )

    return {
        "predicted_mor_label": predicted_mor_label,
        "predicted_label_text": predicted_label_text,
        "mor_probability": round(mor_probability, 6),
        "non_mor_probability": round(non_mor_probability, 6),
        "prediction_confidence": round(float(prediction_confidence), 6),
    }


# ============================================================
# 7. CATEGORY / SUB-CATEGORY PREDICTION
# ============================================================

def predict_category_for_text(report_text: str, category_artifacts: dict) -> dict:
    """
    Predict MOR category/sub-category using trained category models.

    Returns:
        predicted_category_level1
        category_level1_confidence
        predicted_event_type
        event_type_confidence
        predicted_report_title
        report_title_confidence
    """
    text = "" if report_text is None else str(report_text).strip()

    output = {
        "predicted_category_level1": None,
        "category_level1_confidence": None,
        "predicted_event_type": None,
        "event_type_confidence": None,
        "predicted_report_title": None,
        "report_title_confidence": None,
    }

    mapping = {
        "category_level1": (
            "predicted_category_level1",
            "category_level1_confidence",
        ),
        "event_type": (
            "predicted_event_type",
            "event_type_confidence",
        ),
        "report_title": (
            "predicted_report_title",
            "report_title_confidence",
        ),
    }

    for key, artifact in category_artifacts.items():
        if key not in mapping:
            continue

        model = artifact["model"]
        vectorizer = artifact["vectorizer"]

        X = vectorizer.transform([text])

        prediction = model.predict(X)[0]
        confidence = float(model.predict_proba(X).max(axis=1)[0])

        pred_col, conf_col = mapping[key]

        output[pred_col] = prediction
        output[conf_col] = round(confidence, 6)

    return output


# ============================================================
# 8. ML + RULE ENGINE + CATEGORY COMBINATION
# ============================================================

def predict_single_report(
    report_text: str,
    binary_model,
    binary_vectorizer,
    rule_engine: RuleEngine,
    category_artifacts: Optional[dict] = None,
) -> Dict:
    """
    Predict one report using:
        1. Binary MOR ML model
        2. Rule engine
        3. Category/sub-category models

    Category prediction is run if:
        ML predicts MOR
        OR
        Rule engine says MOR
    """
    ml_result = predict_ml_for_text(
        report_text=report_text,
        model=binary_model,
        vectorizer=binary_vectorizer,
    )

    rule_result = rule_engine.evaluate_report(report_text)

    category_result = {
        "predicted_category_level1": None,
        "category_level1_confidence": None,
        "predicted_event_type": None,
        "event_type_confidence": None,
        "predicted_report_title": None,
        "report_title_confidence": None,
    }

    should_predict_category = (
        ml_result["predicted_mor_label"] == 1
        or rule_result["rule_based_mor"] == 1
    )

    if should_predict_category and category_artifacts:
        category_result = predict_category_for_text(
            report_text=report_text,
            category_artifacts=category_artifacts,
        )

    combined_row = pd.Series({
        **ml_result,
        "rule_based_mor": rule_result["rule_based_mor"],
        "rule_confidence": rule_result["rule_confidence"],
    })

    final_decision = combine_ml_and_rule_decision(combined_row).to_dict()

    return {
        "input_report_text": report_text,

        # ML binary output
        **ml_result,

        # Rule engine output
        "rule_based_mor": rule_result["rule_based_mor"],
        "rule_confidence": rule_result["rule_confidence"],
        "matched_rule_count": rule_result["matched_rule_count"],
        "matched_rule_ids": rule_result["matched_rule_ids"],
        "matched_keywords": rule_result["matched_keywords"],
        "matched_domains": rule_result["matched_domains"],
        "matched_subcategories": rule_result["matched_subcategories"],
        "primary_rule_id": rule_result["primary_rule_id"],
        "primary_domain": rule_result["primary_domain"],
        "primary_subcategory": rule_result["primary_subcategory"],
        "primary_condition": rule_result["primary_condition"],
        "rule_review_hint": rule_result["review_hint"],

        # Category/sub-category output
        **category_result,

        # Final combined decision output
        **final_decision,
    }


def predict_dataframe(
    input_df: pd.DataFrame,
    text_column: str,
    binary_model,
    binary_vectorizer,
    rule_engine: RuleEngine,
    category_artifacts: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Apply ML + rule engine + category prediction to a dataframe.
    """
    if text_column not in input_df.columns:
        raise ValueError(
            f"Text column '{text_column}' not found.\n"
            f"Available columns: {list(input_df.columns)}"
        )

    output_df = input_df.copy()

    prediction_rows = []

    for text in output_df[text_column].fillna("").astype(str).tolist():
        prediction_rows.append(
            predict_single_report(
                report_text=text,
                binary_model=binary_model,
                binary_vectorizer=binary_vectorizer,
                rule_engine=rule_engine,
                category_artifacts=category_artifacts,
            )
        )

    prediction_df = pd.DataFrame(prediction_rows)

    # Avoid duplicating text if input already contains clean_report_text
    if "input_report_text" in prediction_df.columns:
        prediction_df = prediction_df.drop(columns=["input_report_text"])

    output_df = pd.concat(
        [
            output_df.reset_index(drop=True),
            prediction_df.reset_index(drop=True),
        ],
        axis=1,
    )

    return output_df


# ============================================================
# 9. SAVE OUTPUT
# ============================================================

def save_predictions(output_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save prediction workbook with:
        1. Full prediction output
        2. Review queue
        3. Prediction summary
    """
    ensure_output_dir(output_path)

    summary_rows = [
        {
            "metric": "total_reports",
            "value": len(output_df),
        },
        {
            "metric": "ml_predicted_mor",
            "value": int((output_df["predicted_mor_label"] == 1).sum()),
        },
        {
            "metric": "ml_predicted_non_mor",
            "value": int((output_df["predicted_mor_label"] == 0).sum()),
        },
        {
            "metric": "rule_based_mor",
            "value": int((output_df["rule_based_mor"] == 1).sum()),
        },
        {
            "metric": "category_prediction_attempted",
            "value": int(output_df["predicted_category_level1"].notna().sum())
            if "predicted_category_level1" in output_df.columns
            else 0,
        },
        {
            "metric": "mandatory_reviews",
            "value": int((output_df["final_review_priority"] == "Mandatory").sum()),
        },
        {
            "metric": "required_reviews",
            "value": int((output_df["final_review_priority"] == "Required").sum()),
        },
        {
            "metric": "recommended_reviews",
            "value": int((output_df["final_review_priority"] == "Recommended").sum()),
        },
        {
            "metric": "no_review",
            "value": int((output_df["final_review_required"] == "No").sum()),
        },
    ]

    summary_df = pd.DataFrame(summary_rows)

    review_queue_df = output_df[
        output_df["final_review_required"] == "Yes"
    ].copy()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        output_df.to_excel(
            writer,
            sheet_name="Predictions_With_Rules",
            index=False,
        )

        review_queue_df.to_excel(
            writer,
            sheet_name="Review_Queue",
            index=False,
        )

        summary_df.to_excel(
            writer,
            sheet_name="Prediction_Summary",
            index=False,
        )


# ============================================================
# 10. COMMAND LINE INTERFACE
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MOR binary ML model + rule engine + category models "
            "on report text or Excel file."
        )
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single report text to classify.",
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Input Excel file path. Defaults to "
            "data/processed/volrep/combined_volrep_2026H1_refined.xlsx"
        ),
    )

    parser.add_argument(
        "--text-column",
        type=str,
        default=DEFAULT_TEXT_COLUMN,
        help="Text column name in Excel input. Default: clean_report_text",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output Excel file path. Defaults to "
            "outputs/predictions/volrep_mor_predictions_with_rules.xlsx"
        ),
    )

    return parser.parse_args()


# ============================================================
# 11. MAIN EXECUTION
# ============================================================

def main() -> None:
    args = parse_args()

    binary_model, binary_vectorizer = load_binary_model_artifacts()
    category_artifacts = load_category_model_artifacts()
    rule_engine = RuleEngine()

    # Mode 1: Single report prediction
    if args.text:
        result = predict_single_report(
            report_text=args.text,
            binary_model=binary_model,
            binary_vectorizer=binary_vectorizer,
            rule_engine=rule_engine,
            category_artifacts=category_artifacts,
        )

        print("\nMOR ML + Rule Engine + Category Prediction")
        print("=" * 60)

        for key, value in result.items():
            print(f"{key}: {value}")

        return

    # Mode 2: Excel batch prediction
    input_path = Path(args.input) if args.input else DEFAULT_INPUT_FILE
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_FILE

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input Excel file not found: {input_path}\n"
            "Provide --input path or ensure refined Volrep file exists."
        )

    input_df = pd.read_excel(input_path, engine="openpyxl")

    output_df = predict_dataframe(
        input_df=input_df,
        text_column=args.text_column,
        binary_model=binary_model,
        binary_vectorizer=binary_vectorizer,
        rule_engine=rule_engine,
        category_artifacts=category_artifacts,
    )

    save_predictions(output_df, output_path)

    print("\nML + Rule Engine + Category prediction completed successfully.")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Rows processed: {len(output_df)}")
    print(f"ML predicted MOR: {(output_df['predicted_mor_label'] == 1).sum()}")
    print(f"Rule-based MOR: {(output_df['rule_based_mor'] == 1).sum()}")
    print(f"Mandatory reviews: {(output_df['final_review_priority'] == 'Mandatory').sum()}")
    print(f"Total review required: {(output_df['final_review_required'] == 'Yes').sum()}")


if __name__ == "__main__":
    main()