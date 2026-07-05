"""
preprocess_occurrence.py

Purpose:
    Pre-process the Occurrence Sheet 2026-MASTER.xlsx file for the MOR Classification project.

Main outputs:
    1. data/processed/occurrence/occurrence_master_cleaned.xlsx
    2. data/processed/training/mor_binary_training_dataset.xlsx
    3. data/processed/training/mor_category_training_dataset.xlsx

Important transformations:
    - Rename BREIF DESCRIPTION to brief_description
    - Create binary mor_label from DGCA REPORTING SCHEME
        MOR -> 1
        VSR -> 0
    - Clean brief_description
    - Normalize REPORT TITLE
    - Normalize CICTT and IATA category labels
    - Remove rows with missing brief_description
    - Handle duplicate summaries using duplicate_group_id and group-based split
    - Add time-based split for chronological validation/testing
    - Create binary training dataset
    - Create MOR-only category training dataset

Run from project root:
    python src/preprocess_occurrence.py
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_OCCURRENCE_DIR = PROJECT_ROOT / "data" / "raw" / "occurrence"
PROCESSED_OCCURRENCE_DIR = PROJECT_ROOT / "data" / "processed" / "occurrence"
TRAINING_DIR = PROJECT_ROOT / "data" / "processed" / "training"

INPUT_FILE = RAW_OCCURRENCE_DIR / "Occurrence Sheet 2026-MASTER.xlsx"
FALLBACK_INPUT_FILE = Path("Occurrence Sheet 2026-MASTER.xlsx")

CLEANED_OUTPUT_FILE = PROCESSED_OCCURRENCE_DIR / "occurrence_master_cleaned.xlsx"
BINARY_TRAINING_FILE = TRAINING_DIR / "mor_binary_training_dataset.xlsx"
CATEGORY_TRAINING_FILE = TRAINING_DIR / "mor_category_training_dataset.xlsx"


# ============================================================
# 2. COLUMN CONFIGURATION
# ============================================================

# Original column name is misspelled in source workbook.
SOURCE_BRIEF_COL = "BREIF DESCRIPTION"
RENAMED_BRIEF_COL = "brief_description"

REQUIRED_COLUMNS = [
    "DGCA REPORTING SCHEME",
    SOURCE_BRIEF_COL,
]

CATEGORY_COLUMNS = [
    "REPORT TITLE",
    "CICTT Category",
    "IATA IDX Parent Level (Level 1)",
    "IATA IDX Parent Level (Level 2)",
    "IATA IDX Event Type (Level 3)",
    "IATA IDX Event Type (Level 4)",
    "SPI Category (Level 1)",
    "SPI Category (Level 2)",
    "Occurence Category",
]

AVIATION_ABBREVIATIONS = {
    r"\bA/C\b": " aircraft ",
    r"\bACFT\b": " aircraft ",
    r"\bRWY\b": " runway ",
    r"\bRNWY\b": " runway ",
    r"\bTWY\b": " taxiway ",
    r"\bPAX\b": " passenger ",
    r"\bENG\b": " engine ",
    r"\bHYD\b": " hydraulic ",
    r"\bFWD\b": " forward ",
    r"\bAFT\b": " aft ",
    r"\bLH\b": " left hand ",
    r"\bRH\b": " right hand ",
    r"\bLHS\b": " left hand side ",
    r"\bRHS\b": " right hand side ",
    r"\bPF\b": " pilot flying ",
    r"\bPM\b": " pilot monitoring ",
    r"\bPIC\b": " pilot in command ",
    r"\bFO\b": " first officer ",
    r"\bCAPT\b": " captain ",
    r"\bATC\b": " air traffic control ",
    r"\bAPU\b": " auxiliary power unit ",
    r"\bGPU\b": " ground power unit ",
    r"\bFOD\b": " foreign object debris ",
    r"\bGA\b": " go around ",
    r"\bGO/A\b": " go around ",
    r"\bRTO\b": " rejected takeoff ",
    r"\bTOGA\b": " takeoff go around ",
}


# ============================================================
# 3. PATH AND BASIC HELPERS
# ============================================================

def resolve_input_path(primary_path: Path, fallback_path: Path) -> Path:
    """Return the first available input path."""
    if primary_path.exists():
        return primary_path
    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(
        f"Occurrence input file not found. Checked:\n"
        f"1. {primary_path}\n"
        f"2. {fallback_path}"
    )


def require_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise an error if required source columns are missing."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Occurrence sheet: {missing}")


def stable_hash_int(text: str) -> int:
    """Return deterministic integer hash for a string."""
    if not isinstance(text, str):
        text = ""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest, 16)


# ============================================================
# 4. TEXT CLEANING AND LABEL NORMALIZATION
# ============================================================

def remove_personal_identifiers(text: object) -> str:
    """
    Remove/mask common personal identifiers from occurrence summaries.
    Conservative approach to preserve aircraft, runway, date, level, speed and event details.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Email addresses and URLs
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Phone numbers: 10+ digit-like strings
    text = re.sub(r"\+?\d[\d\s\-]{9,}\d", " ", text)

    # IDs where explicitly labelled
    text = re.sub(r"\bIGA\s*[-:]?\s*\d{4,8}\b", " IGA_ID ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bSTAFF\s*ID\s*[-:]?\s*\d{3,8}\b", " STAFF_ID ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bEMP(?:LOYEE)?\s*ID\s*[-:]?\s*\d{3,8}\b", " EMPLOYEE_ID ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bPNR\s*[-:]?\s*[A-Z0-9]{5,8}\b", " PNR_ID ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bCRN?\s*(?:NO|NUMBER)?\s*[-:]?\s*\d{5,12}\b", " CASE_ID ", text, flags=re.IGNORECASE)

    # Names following common passenger/person titles
    text = re.sub(r"\bMR\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}", " passenger ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bMS\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}", " passenger ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bMRS\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}", " passenger ", text, flags=re.IGNORECASE)

    return text


def normalize_aviation_terms(text: str) -> str:
    """Normalize aviation abbreviations while preserving key safety meaning."""
    for pattern, replacement in AVIATION_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def clean_text_for_ml(text: object) -> str:
    """Clean brief description into ML-ready text."""
    text = remove_personal_identifiers(text)

    # Remove boilerplate phrases often found in occurrence summaries
    text = re.sub(r"\bas per (crew|report|information|details received)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthis is for your information\b", " ", text, flags=re.IGNORECASE)

    text = normalize_aviation_terms(text)
    text = text.lower()

    # Normalize whitespace and remove unusual characters while keeping useful aviation separators
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^a-z0-9\s/\-.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_duplicate_key(text: object) -> str:
    """
    Create duplicate-detection key from brief description.
    This is stricter than label cleaning and removes punctuation so near-identical summaries group together.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().replace("\u00a0", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_label(value: object, unknown: str = "UNKNOWN") -> str:
    """
    Standardize label values into one clean representation.
    Handles spaces, casing, NBSP, punctuation and common blank values.
    """
    if pd.isna(value):
        return unknown

    text = str(value).replace("\u00a0", " ").strip()
    text = re.sub(r"\s+", " ", text)

    if text == "" or text.upper() in {"NA", "N/A", "NIL", "NONE", "NAN", "-"}:
        return unknown

    # Remove excessive punctuation but preserve slash, hyphen, ampersand for aviation labels
    text = re.sub(r"[^A-Za-z0-9/&()\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Title case for readability; then fix common acronyms.
    text = text.title()
    acronym_replacements = {
        "Atc": "ATC",
        "Tcas": "TCAS",
        "Acas": "ACAS",
        "Gnss": "GNSS",
        "Rfi": "RFI",
        "Fod": "FOD",
        "Gpu": "GPU",
        "Apu": "APU",
        "EcAM": "ECAM",
        "Ecam": "ECAM",
        "Dg": "DG",
        "MOR": "MOR",
        "VSR": "VSR",
    }
    for old, new in acronym_replacements.items():
        text = re.sub(rf"\b{old}\b", new, text)

    return text


def normalize_reporting_scheme(value: object) -> str:
    """Normalize DGCA REPORTING SCHEME to MOR/VSR/UNKNOWN."""
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip().upper().replace(" ", "")
    if text == "MOR":
        return "MOR"
    if text == "VSR":
        return "VSR"
    return "UNKNOWN"


def normalize_report_title(value: object) -> str:
    """Normalize REPORT TITLE specifically and merge common variants."""
    label = clean_label(value)

    # Standardize known variants observed in safety occurrence sheets.
    mapping = {
        "Balked Landing": "Balked Landing",
        "Balked Landing ": "Balked Landing",
        "Bird/Wildlife Strike": "Bird/Wildlife Strike",
        "Bird Wildlife Strike": "Bird/Wildlife Strike",
        "Case Of Wrong Taxi": "Wrong Taxiway / Taxi Deviation",
        "Wrong Turn On Taxiway (Voluntary Information)": "Wrong Taxiway / Taxi Deviation",
        "Ramp Occurrence": "Ramp Occurrence",
        "Ground Turn Back": "Ground Turn Back",
        "GNSS Interference": "GNSS Interference",
        "Unruly": "Unruly Passenger",
    }
    return mapping.get(label, label)


# ============================================================
# 5. SPLIT STRATEGIES
# ============================================================

def assign_group_based_split(duplicate_key: str) -> str:
    """
    Assign deterministic group-based split using normalized brief description.
    Same duplicate/near-duplicate summaries always go to the same split.

    Split ratio approximately:
        train: 70%
        validation: 15%
        test: 15%
    """
    hashed = stable_hash_int(duplicate_key) % 100
    if hashed < 70:
        return "train"
    if hashed < 85:
        return "validation"
    return "test"


def add_time_based_split(df: pd.DataFrame, date_col: str = "date_of_occurrence_parsed") -> pd.DataFrame:
    """
    Add chronological split based on occurrence date.
    Earliest 70% = train, next 15% = validation, latest 15% = test.
    """
    df = df.copy()
    valid_dates = df[date_col].dropna().sort_values()

    if len(valid_dates) == 0:
        df["time_based_split"] = "unknown_date"
        return df

    train_cutoff = valid_dates.quantile(0.70)
    val_cutoff = valid_dates.quantile(0.85)

    def split_date(date_value: pd.Timestamp) -> str:
        if pd.isna(date_value):
            return "unknown_date"
        if date_value <= train_cutoff:
            return "train"
        if date_value <= val_cutoff:
            return "validation"
        return "test"

    df["time_based_split"] = df[date_col].apply(split_date)
    return df


# ============================================================
# 6. MAIN PREPROCESSING PIPELINE
# ============================================================

def load_occurrence_master() -> pd.DataFrame:
    """Load the MASTER sheet from occurrence workbook."""
    input_path = resolve_input_path(INPUT_FILE, FALLBACK_INPUT_FILE)
    df = pd.read_excel(input_path, sheet_name="MASTER", engine="openpyxl")
    require_columns(df, REQUIRED_COLUMNS)
    df["source_file"] = input_path.name
    return df


def preprocess_occurrence_master() -> pd.DataFrame:
    """Clean and enrich Occurrence Master data."""
    df = load_occurrence_master()

    # Keep original report number if available and create internal stable ID.
    df.insert(0, "occurrence_internal_id", range(1, len(df) + 1))

    # Rename typo column.
    df = df.rename(columns={SOURCE_BRIEF_COL: RENAMED_BRIEF_COL})

    # Remove rows with missing brief description.
    df[RENAMED_BRIEF_COL] = df[RENAMED_BRIEF_COL].replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(subset=[RENAMED_BRIEF_COL]).copy()

    # Date parsing.
    if "DATE OF OCCURRENCE" in df.columns:
        df["date_of_occurrence_parsed"] = pd.to_datetime(df["DATE OF OCCURRENCE"], errors="coerce")
    else:
        df["date_of_occurrence_parsed"] = pd.NaT

    # Normalize reporting scheme and create binary target.
    df["dgca_reporting_scheme_normalized"] = df["DGCA REPORTING SCHEME"].apply(normalize_reporting_scheme)
    df = df[df["dgca_reporting_scheme_normalized"].isin(["MOR", "VSR"])].copy()
    df["mor_label"] = df["dgca_reporting_scheme_normalized"].map({"MOR": 1, "VSR": 0}).astype(int)

    # Clean brief description.
    df["clean_brief_description"] = df[RENAMED_BRIEF_COL].apply(clean_text_for_ml)
    df["brief_word_count"] = df["clean_brief_description"].str.split().str.len()

    # Remove rows still empty after cleaning.
    df = df[df["clean_brief_description"].fillna("").str.strip() != ""].copy()

    # Duplicate handling using normalized brief summary.
    df["normalized_brief_for_duplicate_check"] = df[RENAMED_BRIEF_COL].apply(normalize_duplicate_key)
    df["duplicate_group_id"] = df["normalized_brief_for_duplicate_check"].apply(
        lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[:12]
    )
    df["is_duplicate_summary"] = df["normalized_brief_for_duplicate_check"].duplicated(keep=False)
    df["duplicate_group_size"] = df.groupby("duplicate_group_id")["duplicate_group_id"].transform("size")

    # Group-based split avoids same duplicate summary leaking into train/test.
    df["group_based_split"] = df["normalized_brief_for_duplicate_check"].apply(assign_group_based_split)

    # Time-based split supports chronological evaluation.
    df = add_time_based_split(df, date_col="date_of_occurrence_parsed")

    # Normalize report title.
    if "REPORT TITLE" in df.columns:
        df["report_title_normalized"] = df["REPORT TITLE"].apply(normalize_report_title)
    else:
        df["report_title_normalized"] = "UNKNOWN"

    # Normalize CICTT and IATA category labels.
    category_rename_map = {
        "CICTT Category": "cictt_category_normalized",
        "IATA IDX Parent Level (Level 1)": "iata_level_1_normalized",
        "IATA IDX Parent Level (Level 2)": "iata_level_2_normalized",
        "IATA IDX Event Type (Level 3)": "iata_event_level_3_normalized",
        "IATA IDX Event Type (Level 4)": "iata_event_level_4_normalized",
        "SPI Category (Level 1)": "spi_level_1_normalized",
        "SPI Category (Level 2)": "spi_level_2_normalized",
        "Occurence Category": "occurrence_category_normalized",
    }

    for source_col, target_col in category_rename_map.items():
        if source_col in df.columns:
            df[target_col] = df[source_col].apply(clean_label)
        else:
            df[target_col] = "UNKNOWN"

    # Correct common spelling inconsistencies.
    df["iata_level_2_normalized"] = df["iata_level_2_normalized"].replace({
        "Flight Path Mangement": "Flight Path Management",
    })

    # Place important ML columns first.
    preferred_cols = [
        "occurrence_internal_id",
        "REPORT NO.",
        "source_file",
        "DATE OF OCCURRENCE",
        "date_of_occurrence_parsed",
        "MONTH",
        "REPORT TYPE",
        "DGCA REPORTING SCHEME",
        "dgca_reporting_scheme_normalized",
        "mor_label",
        "brief_description",
        "clean_brief_description",
        "brief_word_count",
        "normalized_brief_for_duplicate_check",
        "duplicate_group_id",
        "is_duplicate_summary",
        "duplicate_group_size",
        "group_based_split",
        "time_based_split",
        "REPORT TITLE",
        "report_title_normalized",
        "CICTT Category",
        "cictt_category_normalized",
        "IATA IDX Parent Level (Level 1)",
        "iata_level_1_normalized",
        "IATA IDX Parent Level (Level 2)",
        "iata_level_2_normalized",
        "IATA IDX Event Type (Level 3)",
        "iata_event_level_3_normalized",
        "IATA IDX Event Type (Level 4)",
        "iata_event_level_4_normalized",
        "SPI Category (Level 1)",
        "spi_level_1_normalized",
        "SPI Category (Level 2)",
        "spi_level_2_normalized",
        "Occurence Category",
        "occurrence_category_normalized",
        "AIRCRAFT TYPE",
        "FLIGHT NO",
        "SECTOR",
        "AIRPORT",
        "PHASE OF FLIGHT",
        "EVENT RISK CLASSIFICATION (ERC)",
    ]
    existing_preferred_cols = [col for col in preferred_cols if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in existing_preferred_cols]
    df = df[existing_preferred_cols + remaining_cols]

    return df


def create_binary_training_dataset(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Create binary MOR/Non-MOR training dataset."""
    cols = [
        "occurrence_internal_id",
        "REPORT NO.",
        "clean_brief_description",
        "brief_description",
        "mor_label",
        "dgca_reporting_scheme_normalized",
        "group_based_split",
        "time_based_split",
        "duplicate_group_id",
        "is_duplicate_summary",
        "duplicate_group_size",
        "REPORT TYPE",
        "date_of_occurrence_parsed",
        "MONTH",
        "AIRCRAFT TYPE",
        "SECTOR",
        "AIRPORT",
        "PHASE OF FLIGHT",
    ]
    cols = [col for col in cols if col in cleaned_df.columns]
    return cleaned_df[cols].copy()


def create_mor_category_training_dataset(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Create MOR-only category/sub-category training dataset."""
    mor_df = cleaned_df[cleaned_df["mor_label"] == 1].copy()

    cols = [
        "occurrence_internal_id",
        "REPORT NO.",
        "clean_brief_description",
        "brief_description",
        "mor_label",
        "dgca_reporting_scheme_normalized",
        "group_based_split",
        "time_based_split",
        "duplicate_group_id",
        "is_duplicate_summary",
        "duplicate_group_size",
        "report_title_normalized",
        "cictt_category_normalized",
        "iata_level_1_normalized",
        "iata_level_2_normalized",
        "iata_event_level_3_normalized",
        "iata_event_level_4_normalized",
        "spi_level_1_normalized",
        "spi_level_2_normalized",
        "occurrence_category_normalized",
        "REPORT TYPE",
        "date_of_occurrence_parsed",
        "MONTH",
        "AIRCRAFT TYPE",
        "SECTOR",
        "AIRPORT",
        "PHASE OF FLIGHT",
        "EVENT RISK CLASSIFICATION (ERC)",
    ]
    cols = [col for col in cols if col in mor_df.columns]
    return mor_df[cols].copy()


def create_quality_summary(cleaned_df: pd.DataFrame, binary_df: pd.DataFrame, mor_category_df: pd.DataFrame) -> pd.DataFrame:
    """Create summary metrics for data quality and outputs."""
    return pd.DataFrame([
        {"metric": "cleaned_occurrence_rows", "value": len(cleaned_df)},
        {"metric": "binary_training_rows", "value": len(binary_df)},
        {"metric": "mor_category_training_rows", "value": len(mor_category_df)},
        {"metric": "mor_rows", "value": int((cleaned_df["mor_label"] == 1).sum())},
        {"metric": "vsr_rows", "value": int((cleaned_df["mor_label"] == 0).sum())},
        {"metric": "missing_clean_text_rows", "value": int(cleaned_df["clean_brief_description"].fillna("").str.strip().eq("").sum())},
        {"metric": "duplicate_summary_rows", "value": int(cleaned_df["is_duplicate_summary"].sum())},
        {"metric": "duplicate_summary_groups", "value": int(cleaned_df.loc[cleaned_df["is_duplicate_summary"], "duplicate_group_id"].nunique())},
        {"metric": "avg_brief_word_count", "value": round(float(cleaned_df["brief_word_count"].mean()), 2)},
        {"metric": "median_brief_word_count", "value": round(float(cleaned_df["brief_word_count"].median()), 2)},
    ])


def create_data_dictionary() -> pd.DataFrame:
    """Create data dictionary for generated columns."""
    rows = [
        ("brief_description", "Renamed version of source column BREIF DESCRIPTION."),
        ("clean_brief_description", "Cleaned ML-ready version of brief_description."),
        ("dgca_reporting_scheme_normalized", "Normalized MOR/VSR scheme."),
        ("mor_label", "Binary target: MOR = 1, VSR = 0."),
        ("report_title_normalized", "Cleaned and standardized report title."),
        ("cictt_category_normalized", "Cleaned CICTT category label."),
        ("iata_level_1_normalized", "Cleaned IATA IDX Parent Level 1 label."),
        ("iata_level_2_normalized", "Cleaned IATA IDX Parent Level 2 label."),
        ("iata_event_level_3_normalized", "Cleaned IATA IDX Event Type Level 3 label."),
        ("iata_event_level_4_normalized", "Cleaned IATA IDX Event Type Level 4 label."),
        ("duplicate_group_id", "Hash ID generated from normalized brief text to group duplicate/near-duplicate summaries."),
        ("is_duplicate_summary", "True when the same normalized brief appears more than once."),
        ("group_based_split", "Deterministic train/validation/test split by duplicate_group_id to avoid duplicate leakage."),
        ("time_based_split", "Chronological train/validation/test split based on DATE OF OCCURRENCE."),
    ]
    return pd.DataFrame(rows, columns=["column", "description"])


def save_outputs(cleaned_df: pd.DataFrame, binary_df: pd.DataFrame, mor_category_df: pd.DataFrame) -> None:
    """Save all output workbooks."""
    PROCESSED_OCCURRENCE_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    quality_summary = create_quality_summary(cleaned_df, binary_df, mor_category_df)
    data_dictionary = create_data_dictionary()

    # Full cleaned occurrence workbook.
    with pd.ExcelWriter(CLEANED_OUTPUT_FILE, engine="openpyxl") as writer:
        cleaned_df.to_excel(writer, sheet_name="Cleaned_Occurrence", index=False)
        quality_summary.to_excel(writer, sheet_name="Quality_Summary", index=False)
        data_dictionary.to_excel(writer, sheet_name="Data_Dictionary", index=False)

    # Binary training dataset.
    with pd.ExcelWriter(BINARY_TRAINING_FILE, engine="openpyxl") as writer:
        binary_df.to_excel(writer, sheet_name="Binary_Training", index=False)
        quality_summary.to_excel(writer, sheet_name="Quality_Summary", index=False)
        data_dictionary.to_excel(writer, sheet_name="Data_Dictionary", index=False)

    # MOR-only category training dataset.
    with pd.ExcelWriter(CATEGORY_TRAINING_FILE, engine="openpyxl") as writer:
        mor_category_df.to_excel(writer, sheet_name="MOR_Category_Training", index=False)
        quality_summary.to_excel(writer, sheet_name="Quality_Summary", index=False)
        data_dictionary.to_excel(writer, sheet_name="Data_Dictionary", index=False)


def main() -> None:
    cleaned_df = preprocess_occurrence_master()
    binary_df = create_binary_training_dataset(cleaned_df)
    mor_category_df = create_mor_category_training_dataset(cleaned_df)
    save_outputs(cleaned_df, binary_df, mor_category_df)

    quality_summary = create_quality_summary(cleaned_df, binary_df, mor_category_df)

    print("Occurrence preprocessing completed successfully.")
    print(f"Cleaned rows: {len(cleaned_df)}")
    print(f"Binary training rows: {len(binary_df)}")
    print(f"MOR category training rows: {len(mor_category_df)}")
    print(f"Cleaned output saved to: {CLEANED_OUTPUT_FILE}")
    print(f"Binary training file saved to: {BINARY_TRAINING_FILE}")
    print(f"MOR category training file saved to: {CATEGORY_TRAINING_FILE}")
    print("\nQuality summary:")
    print(quality_summary.to_string(index=False))


if __name__ == "__main__":
    main()
