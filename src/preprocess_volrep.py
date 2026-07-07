"""
preprocess_volrep.py

Purpose:
    Pre-process Volrep pilot/crew safety report Excel files for the MOR Classification project.

Input files expected:
    data/raw/volrep/Volrep_Processed (Jan-Mar).xlsx
    data/raw/volrep/Volrep_Processed (APR-JUN).xlsx

Output file created:
    data/processed/volrep/combined_volrep_2026H1_cleaned.xlsx

If you run this script from a folder where the Excel files are present directly,
it will also work because fallback paths are included.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_VOLREP_DIR = PROJECT_ROOT / "data" / "raw" / "volrep"
PROCESSED_VOLREP_DIR = PROJECT_ROOT / "data" / "processed" / "volrep"

JAN_MAR_FILE = RAW_VOLREP_DIR / "Volrep_Processed (Jan-Mar).xlsx"
APR_JUN_FILE = RAW_VOLREP_DIR / "Volrep_Processed (APR-JUN).xlsx"

OUTPUT_FILE = PROCESSED_VOLREP_DIR / "combined_volrep_2026H1_cleaned.xlsx"

# Fallback paths: useful if you temporarily run the script in the same folder as the Excel files.
FALLBACK_JAN_MAR_FILE = Path("Volrep_Processed (Jan-Mar).xlsx")
FALLBACK_APR_JUN_FILE = Path("Volrep_Processed (APR-JUN).xlsx")


# ============================================================
# 2. COLUMN CONFIGURATION
# ============================================================

# These are the possible text columns observed across both Volrep Excel files.
# The script coalesces them into one final raw_report_text column.
EVENT_DESCRIPTION_COLUMNS = [
    "Event Description - Please give an account of what took place and how/why , also describing how you managed the event",
    "Description of Event (include sequence, contributing factors, and crew actions)",
    "Event Description",
    "Description",
    "Other",
]

DATE_COLUMNS = [
    "Date and Time of Occurrence (24Hr format and UTC)",
    "Date and Time of Occurrence (24Hr format)",
    "Date & time of event",
    "Date_2",
]

STATION_COLUMNS = [
    "Departure Station",
    "Airfield of Landing",
    "Arrival station",
    "Place of Occurrence",
    "Place of Occurrence_2",
    "Location",
]


# ============================================================
# 3. GENERIC HELPERS
# ============================================================

def resolve_input_path(primary_path: Path, fallback_path: Path) -> Path:
    """Return the available path for an input file."""
    if primary_path.exists():
        return primary_path
    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(
        f"Input file not found at either location:\n"
        f"1. {primary_path}\n"
        f"2. {fallback_path}\n"
    )


def standardize_column_name(col: str) -> str:
    """Convert column names to a clean snake_case format."""
    col = str(col).strip()
    col = col.replace("/", "_")
    col = col.replace("&", "and")
    col = re.sub(r"[^A-Za-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_").lower()


def first_non_empty(row: pd.Series, columns: Iterable[str]) -> Optional[str]:
    """Return the first non-empty value from a list of possible columns."""
    for col in columns:
        if col in row.index:
            value = row[col]
            if pd.notna(value) and str(value).strip() != "":
                return str(value).strip()
    return np.nan


def parse_date_from_row(row: pd.Series) -> pd.Timestamp:
    """Coalesce date columns and parse them into pandas datetime."""
    raw_date = first_non_empty(row, DATE_COLUMNS)
    return pd.to_datetime(raw_date, errors="coerce")


# ============================================================
# 4. NORMALIZATION FUNCTIONS
# ============================================================

def normalize_aircraft_type(value: object) -> str:
    """Normalize aircraft type variants into standard aircraft families."""
    if pd.isna(value):
        return "UNKNOWN"

    text = str(value).upper().strip()
    text = re.sub(r"\s+", "", text)

    if text in {"", "NAN", "NONE", "NA"}:
        return "UNKNOWN"

    # Airbus narrow-body variants
    if text.startswith("A320"):
        return "A320"
    if text.startswith("A321"):
        return "A321"
    if text.startswith("A319"):
        return "A319"

    # ATR variants
    if text.startswith("ATR") or "ATR72" in text or "ATR-72" in text:
        return "ATR"

    # Boeing variants, if they appear in future data
    if text.startswith("B777") or text.startswith("777"):
        return "B777"
    if text.startswith("B787") or text.startswith("787"):
        return "B787"
    if text.startswith("B738") or text.startswith("737") or text.startswith("B737"):
        return "B737"

    return text


def normalize_phase_of_flight(value: object) -> str:
    """Normalize phase of flight into controlled labels."""
    if pd.isna(value):
        return "UNKNOWN"

    text = str(value).upper().strip()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text in {"", "NAN", "NA", "NONE"}:
        return "UNKNOWN"

    phase_map = {
        "LANDING": "LANDING",
        "APPROACH": "APPROACH",
        "TAKE OFF": "TAKEOFF",
        "TAKEOFF": "TAKEOFF",
        "TAKE OFF RUN": "TAKEOFF_RUN",
        "TAKEOFF RUN": "TAKEOFF_RUN",
        "TAXI OUT": "TAXI_OUT",
        "TAXI IN": "TAXI_IN",
        "PUSH BACK": "PUSHBACK_START",
        "PUSHBACK": "PUSHBACK_START",
        "PUSHBACK START": "PUSHBACK_START",
        "MISSED APPROACH": "MISSED_APPROACH",
        "GO AROUND": "GO_AROUND",
        "BALKED LANDING": "BALKED_LANDING",
        "CLIMB": "CLIMB",
        "INITIAL CLIMB": "INITIAL_CLIMB",
        "CRUISE": "CRUISE",
        "DESCENT": "DESCENT",
        "PARKED": "PARKING_GATE",
        "PARKING": "PARKING_GATE",
        "GROUND": "GROUND",
        "ON GROUND DOOR CLOSE WITH ENGINES RUNNING": "GROUND_ENGINE_RUNNING",
    }

    if text in phase_map:
        return phase_map[text]

    # Fuzzy fallback rules
    if "GO" in text and "AROUND" in text:
        return "GO_AROUND"
    if "MISSED" in text and "APPROACH" in text:
        return "MISSED_APPROACH"
    if "TAXI" in text and "OUT" in text:
        return "TAXI_OUT"
    if "TAXI" in text and "IN" in text:
        return "TAXI_IN"
    if "TAKE" in text and "OFF" in text:
        return "TAKEOFF"
    if "LAND" in text:
        return "LANDING"
    if "APPROACH" in text:
        return "APPROACH"

    return text.replace(" ", "_")


def normalize_station_name(value: object) -> str:
    """
    Normalize airport/station fields.

    Examples:
        "VEDG : DURGAPUR AIRPORT" -> "VEDG - DURGAPUR AIRPORT"
        "VIDP : DELHI,INDIRA GANDHI INTL" -> "VIDP - DELHI INDIRA GANDHI INTL"
    """
    if pd.isna(value):
        return "UNKNOWN"

    text = str(value).upper().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u00a0", " ")

    if text in {"", "NAN", "NA", "NONE", "OTHERS", "OTHER"}:
        return "UNKNOWN"

    # Remove repeated punctuation and normalize separators
    text = text.replace(";", ",")
    text = re.sub(r"\s*,\s*", " ", text)
    text = re.sub(r"\s*:\s*", " : ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # ICAO code format: XXXX : AIRPORT NAME
    match = re.match(r"^([A-Z]{4})\s*:\s*(.+)$", text)
    if match:
        code = match.group(1).strip()
        name = match.group(2).strip()
        name = re.sub(r"\s+", " ", name)
        return f"{code} - {name}"

    return text


def extract_station_code(normalized_station: str) -> str:
    """Extract ICAO station code from normalized station text, if available."""
    if not isinstance(normalized_station, str):
        return "UNKNOWN"
    match = re.match(r"^([A-Z]{4})\s*-", normalized_station)
    if match:
        return match.group(1)
    return "UNKNOWN"


# ============================================================
# 5. TEXT DE-IDENTIFICATION AND CLEANING
# ============================================================

def remove_personal_identifiers(text: object) -> str:
    """
    Mask/remove common personal identifiers from pilot reports.

    This is intentionally conservative: aviation-critical numbers such as V1,
    FL370, RWY 27, altitude, speed, etc. should not be removed blindly.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " ", text)

    # Phone numbers: 10+ continuous digits or country-code style numbers
    text = re.sub(r"\+?\d[\d\s\-]{9,}\d", " ", text)

    # Employee / staff identifiers commonly seen in reports
    text = re.sub(r"\bIGA\s*[-:]?\s*\d{4,8}\b", " IGA_ID ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bSTAFF\s*ID\s*[-:]?\s*\d{3,8}\b", " STAFF_ID ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bEMP(?:LOYEE)?\s*ID\s*[-:]?\s*\d{3,8}\b", " EMPLOYEE_ID ", text, flags=re.IGNORECASE)

    # PNR / CRN-like values where explicitly labelled
    text = re.sub(r"\bPNR\s*[-:]?\s*[A-Z0-9]{5,8}\b", " PNR_ID ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bCRN?\s*(?:NO|NUMBER)?\s*[-:]?\s*\d{5,12}\b", " CASE_ID ", text, flags=re.IGNORECASE)

    # Names following operational titles. We preserve the role, remove the name.
    text = re.sub(r"\bCAPT(?:AIN)?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}", " Captain ", text)
    text = re.sub(r"\bFO\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}", " First Officer ", text)
    text = re.sub(r"\bMR\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}", " passenger ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bMS\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}", " passenger ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bMRS\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}", " passenger ", text, flags=re.IGNORECASE)

    return text


def normalize_aviation_terms(text: str) -> str:
    """Normalize common aviation abbreviations without removing important safety terms."""
    replacements = {
        r"\bA/C\b": " aircraft ",
        r"\bACFT\b": " aircraft ",
        r"\bAEROPLANE\b": " aircraft ",
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
        r"\bAP\b": " autopilot ",
        r"\bA/P\b": " autopilot ",
        r"\bATC\b": " air traffic control ",
        r"\bAPU\b": " auxiliary power unit ",
        r"\bGPU\b": " ground power unit ",
        r"\bFOD\b": " foreign object debris ",
        r"\bGA\b": " go around ",
        r"\bGO/A\b": " go around ",
        r"\bRTO\b": " rejected takeoff ",
        r"\bTOGA\b": " takeoff go around ",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def clean_natural_language_text(text: object) -> str:
    """Clean report narrative text for ML while preserving aviation meaning."""
    text = remove_personal_identifiers(text)

    # Remove common email/report greetings and sign-offs
    text = re.sub(r"\b(dear team|good morning|good evening|greetings|dear sir|dear madam)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(kind regards|warm regards|thanks and regards|thank you|regards)\b", " ", text, flags=re.IGNORECASE)

    # Normalize aviation abbreviations before lowercasing
    text = normalize_aviation_terms(text)

    # Lowercase for ML consistency
    text = text.lower()

    # Replace line breaks/tabs with spaces
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Keep letters, numbers, and selected separators useful in aviation context
    text = re.sub(r"[^a-z0-9\s/\-.]", " ", text)

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 6. MAIN PREPROCESSING PIPELINE
# ============================================================

def load_volrep_file(file_path: Path, source_period: str) -> pd.DataFrame:
    """Load one Volrep file and add source tracking columns."""
    df = pd.read_excel(file_path, sheet_name="Sheet1", engine="openpyxl")
    df["source_file"] = file_path.name
    df["source_period"] = source_period
    return df


def preprocess_volrep_files() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Combine and preprocess Volrep Jan-Mar and Apr-Jun files."""

    jan_file = resolve_input_path(JAN_MAR_FILE, FALLBACK_JAN_MAR_FILE)
    apr_file = resolve_input_path(APR_JUN_FILE, FALLBACK_APR_JUN_FILE)

    jan_df = load_volrep_file(jan_file, "Jan-Mar")
    apr_df = load_volrep_file(apr_file, "Apr-Jun")

    combined_df = pd.concat([jan_df, apr_df], ignore_index=True, sort=False)

    # Preserve original row ordering and create a stable internal ID
    combined_df.insert(0, "volrep_internal_id", range(1, len(combined_df) + 1))

    # Main coalesced report narrative
    combined_df["raw_report_text"] = combined_df.apply(
        lambda row: first_non_empty(row, EVENT_DESCRIPTION_COLUMNS), axis=1
    )

    # Parsed event date
    combined_df["event_datetime_parsed"] = combined_df.apply(parse_date_from_row, axis=1)
    combined_df["event_date"] = combined_df["event_datetime_parsed"].dt.date

    # Normalized aircraft type
    if "Aircraft Type" in combined_df.columns:
        combined_df["aircraft_type_normalized"] = combined_df["Aircraft Type"].apply(normalize_aircraft_type)
    elif "Aircraft Type (if applicable)" in combined_df.columns:
        combined_df["aircraft_type_normalized"] = combined_df["Aircraft Type (if applicable)"].apply(normalize_aircraft_type)
    else:
        combined_df["aircraft_type_normalized"] = "UNKNOWN"

    # Normalized phase of flight
    if "Phase of flight (Tick as applicable)" in combined_df.columns:
        combined_df["phase_of_flight_normalized"] = combined_df["Phase of flight (Tick as applicable)"].apply(normalize_phase_of_flight)
    elif "Phase of Flight" in combined_df.columns:
        combined_df["phase_of_flight_normalized"] = combined_df["Phase of Flight"].apply(normalize_phase_of_flight)
    else:
        combined_df["phase_of_flight_normalized"] = "UNKNOWN"

    # Normalize station columns and extract station codes
    for col in STATION_COLUMNS:
        if col in combined_df.columns:
            normalized_col = standardize_column_name(col) + "_normalized"
            code_col = standardize_column_name(col) + "_code"
            combined_df[normalized_col] = combined_df[col].apply(normalize_station_name)
            combined_df[code_col] = combined_df[normalized_col].apply(extract_station_code)

    # Cleaned ML-ready text
    combined_df["clean_report_text"] = combined_df["raw_report_text"].apply(clean_natural_language_text)

    # Basic quality flags
    combined_df["raw_text_word_count"] = combined_df["raw_report_text"].fillna("").astype(str).str.split().str.len()
    combined_df["clean_text_word_count"] = combined_df["clean_report_text"].fillna("").astype(str).str.split().str.len()
    combined_df["is_text_missing"] = combined_df["clean_report_text"].fillna("").str.strip().eq("")
    combined_df["is_short_text"] = combined_df["clean_text_word_count"] < 10
    combined_df["duplicate_clean_text"] = combined_df["clean_report_text"].duplicated(keep=False)

    # Suggested final ML/inference schema columns first
    preferred_cols = [
        "volrep_internal_id",
        "Number",
        "source_file",
        "source_period",
        "Reportee Role",
        "aircraft_type_normalized",
        "phase_of_flight_normalized",
        "event_datetime_parsed",
        "event_date",
        "raw_report_text",
        "clean_report_text",
        "raw_text_word_count",
        "clean_text_word_count",
        "is_text_missing",
        "is_short_text",
        "duplicate_clean_text",
    ]

    existing_preferred_cols = [c for c in preferred_cols if c in combined_df.columns]
    remaining_cols = [c for c in combined_df.columns if c not in existing_preferred_cols]
    combined_df = combined_df[existing_preferred_cols + remaining_cols]

    # Quality summary
    quality_summary = pd.DataFrame(
        [
            {"metric": "total_rows", "value": len(combined_df)},
            {"metric": "jan_mar_rows", "value": int((combined_df["source_period"] == "Jan-Mar").sum())},
            {"metric": "apr_jun_rows", "value": int((combined_df["source_period"] == "Apr-Jun").sum())},
            {"metric": "missing_clean_text_rows", "value": int(combined_df["is_text_missing"].sum())},
            {"metric": "short_text_rows_lt_10_words", "value": int(combined_df["is_short_text"].sum())},
            {"metric": "duplicate_clean_text_rows", "value": int(combined_df["duplicate_clean_text"].sum())},
            {"metric": "avg_clean_text_word_count", "value": round(float(combined_df["clean_text_word_count"].mean()), 2)},
            {"metric": "median_clean_text_word_count", "value": round(float(combined_df["clean_text_word_count"].median()), 2)},
            {"metric": "parsed_event_datetime_rows", "value": int(combined_df["event_datetime_parsed"].notna().sum())},
        ]
    )

    return combined_df, quality_summary


def save_output(cleaned_df: pd.DataFrame, quality_summary: pd.DataFrame) -> None:
    """Save cleaned output workbook with multiple useful sheets."""
    PROCESSED_VOLREP_DIR.mkdir(parents=True, exist_ok=True)

    data_dictionary = pd.DataFrame(
        [
            {"column": "raw_report_text", "description": "First non-empty report narrative coalesced from all possible event-description columns."},
            {"column": "clean_report_text", "description": "De-identified and normalized text suitable for ML inference/training."},
            {"column": "phase_of_flight_normalized", "description": "Controlled phase-of-flight label."},
            {"column": "aircraft_type_normalized", "description": "Standardized aircraft family/type."},
            {"column": "*_normalized", "description": "Normalized station text, usually CODE - STATION NAME."},
            {"column": "*_code", "description": "Extracted ICAO station code where available."},
            {"column": "is_short_text", "description": "True if cleaned text has fewer than 10 words; should be manually reviewed."},
            {"column": "duplicate_clean_text", "description": "True if another row has exactly the same cleaned text."},
        ]
    )

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        cleaned_df.to_excel(writer, sheet_name="Cleaned_Volrep", index=False)
        quality_summary.to_excel(writer, sheet_name="Quality_Summary", index=False)
        data_dictionary.to_excel(writer, sheet_name="Data_Dictionary", index=False)


def main() -> None:
    cleaned_df, quality_summary = preprocess_volrep_files()
    save_output(cleaned_df, quality_summary)

    print("Volrep preprocessing completed successfully.")
    print(f"Rows processed: {len(cleaned_df)}")
    print(f"Output saved to: {OUTPUT_FILE}")
    print("\nQuality summary:")
    print(quality_summary.to_string(index=False))


if __name__ == "__main__":
    main()
