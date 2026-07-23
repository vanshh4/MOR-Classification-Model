"""
Streamlit UI for the IndiGo Flight Safety MOR Classification Model.

Store at:
    app/streamlit_app.py

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Project paths and imports
# -----------------------------------------------------------------------------


def find_project_root() -> Path:
    current_file = Path(__file__).resolve()
    if current_file.parent.name.lower() == "app":
        return current_file.parents[1]
    for parent in [current_file.parent, *current_file.parents]:
        if (parent / "src").exists() and (parent / "models").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
METADATA_FILE = PROJECT_ROOT / "models" / "model_metadata.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import predict_mor_with_rules as predictor  # type: ignore
    from rule_engine import RuleEngine  # type: ignore

    if hasattr(predictor, "load_binary_model_artifacts"):
        load_binary_model_artifacts = predictor.load_binary_model_artifacts
    elif hasattr(predictor, "load_model_artifacts"):
        load_binary_model_artifacts = predictor.load_model_artifacts
    else:
        raise ImportError("Binary model loader was not found in predict_mor_with_rules.py")

    load_category_model_artifacts = predictor.load_category_model_artifacts
    predict_single_report = predictor.predict_single_report
except Exception as import_error:
    st.error("The prediction modules could not be imported. Check the files in src/.")
    st.exception(import_error)
    st.stop()


# -----------------------------------------------------------------------------
# Page setup and state
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Flight Safety MOR Classifier",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

EXAMPLE_REPORT = (
    "During initial climb, the aircraft experienced a suspected bird strike. "
    "Aircraft parameters remained normal and engineering inspection was "
    "requested after landing."
)

DEFAULT_METADATA = {
    "binary_model_version": "1.1",
    "category_model_version": "1.0",
    "mor_threshold": 0.45,
    "training_dataset_version": "Occurrence Master 2026",
    "trained_on": "Not recorded",
}

for key, value in {
    "report_text": "",
    "prediction_result": None,
    "prediction_completed": False,
    "input_message": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clear_report() -> None:
    st.session_state.report_text = ""
    st.session_state.prediction_result = None
    st.session_state.prediction_completed = False
    st.session_state.input_message = ""


def load_example() -> None:
    st.session_state.report_text = EXAMPLE_REPORT
    st.session_state.prediction_result = None
    st.session_state.prediction_completed = False
    st.session_state.input_message = ""


def load_metadata() -> Dict[str, Any]:
    metadata = DEFAULT_METADATA.copy()
    if METADATA_FILE.exists():
        try:
            loaded = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass
    return metadata


MODEL_METADATA = load_metadata()
MODEL_VERSION = str(MODEL_METADATA.get("binary_model_version", "1.1"))
MOR_THRESHOLD = float(MODEL_METADATA.get("mor_threshold", 0.45))


# -----------------------------------------------------------------------------
# Visual design
# -----------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #071C4D;
        --blue: #173E8F;
        --action: #276EF1;
        --page: #F4F7FB;
        --card: #FFFFFF;
        --input: #F1F4F8;
        --text: #101828;
        --muted: #667085;
        --border: #D7DFEA;
        --critical: #B42318;
        --safe: #027A48;
        --review: #B54708;
        --info: #175CD3;
    }

    html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", Arial, sans-serif;
    }

    .stApp { background: var(--page); color: var(--text); }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { max-width: 1220px; padding-top: 0; padding-bottom: 2rem; }

    .navbar-shell {
        width: 100vw;
        position: relative;
        left: 50%;
        right: 50%;
        margin-left: -50vw;
        margin-right: -50vw;
        background: #FFFFFF;
        border-bottom: 1px solid var(--border);
        margin-bottom: 22px;
    }
    .navbar {
        max-width: 1220px;
        margin: 0 auto;
        min-height: 66px;
        padding: 0 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
    }
    .brand-area { display: flex; align-items: center; gap: 12px; }
    .brand-mark {
        width: 38px; height: 38px; border-radius: 8px;
        display: grid; place-items: center;
        background: var(--navy); color: #FFFFFF;
        font-size: 14px; font-weight: 800;
    }
    .brand-title { color: var(--navy); font-size: 16px; font-weight: 800; }
    .brand-subtitle { color: var(--muted); font-size: 12px; margin-top: 2px; }
    .nav-flow { display: flex; align-items: center; gap: 22px; flex-wrap: wrap; }
    .nav-flow span { color: #475467; font-size: 13px; font-weight: 600; }
    .nav-meta { color: var(--blue); font-size: 12px; font-weight: 700; white-space: nowrap; }

    .intro {
        background: #EDF2F8;
        border: 1px solid var(--border);
        border-left: 4px solid var(--blue);
        padding: 22px 24px;
        margin-bottom: 24px;
    }
    .intro h1 { margin: 0 0 8px; color: var(--navy); font-size: 30px; line-height: 1.2; }
    .intro p { margin: 0; color: #475467; font-size: 15px; line-height: 1.65; }
    .intro-note { margin-top: 8px !important; font-size: 13px !important; color: var(--muted) !important; }

    .status-line {
        display: flex; align-items: center; gap: 8px;
        color: #475467; font-size: 13px; margin-bottom: 22px;
    }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--safe); display: inline-block; }

    .section-title { color: var(--navy); font-size: 19px; font-weight: 800; margin: 0 0 6px; }
    .section-copy { color: var(--muted); font-size: 14px; line-height: 1.55; margin-bottom: 12px; }
    .privacy-note { color: #475467; font-size: 12px; line-height: 1.5; margin-top: 8px; }

    textarea {
        background: var(--input) !important;
        color: var(--text) !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        padding: 14px !important;
        font-size: 15px !important;
        line-height: 1.55 !important;
        caret-color: var(--action) !important;
    }
    textarea::placeholder { color: #7A879A !important; opacity: 1 !important; }
    textarea:focus {
        border-color: var(--action) !important;
        box-shadow: 0 0 0 2px rgba(39,110,241,.14) !important;
    }
    div[data-baseweb="textarea"] { background: var(--input) !important; border-radius: 10px !important; }

    div.stButton > button {
        border-radius: 8px;
        min-height: 42px;
        font-weight: 700;
        border: 1px solid var(--border);
    }
    div[data-testid="column"]:first-of-type div.stButton > button {
        background: var(--action); color: #FFFFFF; border-color: var(--action);
    }
    div.stButton > button:hover { border-color: var(--action); color: var(--action); }
    div[data-testid="column"]:first-of-type div.stButton > button:hover {
        background: #1E5ED4; color: #FFFFFF; border-color: #1E5ED4;
    }

    .count-line { color: var(--muted); font-size: 12px; margin: 5px 0 12px; }

    .result-grid {
        display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px; margin: 14px 0;
    }
    .result-card {
        background: var(--card); border: 1px solid var(--border);
        border-left: 4px solid var(--info); padding: 15px 16px; min-height: 106px;
    }
    .result-card.critical { border-left-color: var(--critical); }
    .result-card.safe { border-left-color: var(--safe); }
    .result-card.review { border-left-color: var(--review); }
    .card-label { color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; }
    .card-value { color: var(--text); font-size: 23px; font-weight: 800; margin-top: 8px; line-height: 1.2; word-break: break-word; }
    .card-note { color: var(--muted); font-size: 12px; margin-top: 7px; line-height: 1.4; }
    .critical-text { color: var(--critical); }
    .safe-text { color: var(--safe); }
    .review-text { color: var(--review); }

    .confidence-track { height: 7px; background: #E7ECF3; margin-top: 10px; overflow: hidden; }
    .confidence-fill { height: 100%; background: var(--action); }

    .review-banner {
        background: #FFFFFF; border: 1px solid var(--border);
        border-left: 5px solid var(--safe); padding: 15px 17px; margin: 12px 0 20px;
    }
    .review-banner.mandatory { border-left-color: var(--critical); background: #FFF8F7; }
    .review-banner.required, .review-banner.recommended { border-left-color: var(--review); background: #FFFCF5; }
    .review-title { color: var(--text); font-size: 17px; font-weight: 800; }
    .review-reason { color: #475467; font-size: 13px; line-height: 1.5; margin-top: 4px; }

    .decision-strip {
        display: grid; grid-template-columns: repeat(3, 1fr);
        background: #FFFFFF; border: 1px solid var(--border); margin-bottom: 20px;
    }
    .decision-cell { padding: 14px 16px; border-right: 1px solid var(--border); }
    .decision-cell:last-child { border-right: 0; }
    .decision-label { color: var(--muted); font-size: 12px; font-weight: 700; }
    .decision-value { color: var(--text); font-size: 15px; font-weight: 800; margin-top: 4px; }

    .detail-panel { background: var(--card); border: 1px solid var(--border); padding: 18px; height: 100%; }
    .detail-heading { color: var(--navy); font-size: 18px; font-weight: 800; margin-bottom: 12px; }
    .detail-row { display: grid; grid-template-columns: 42% 58%; gap: 12px; padding: 9px 0; border-bottom: 1px solid #E5EAF2; }
    .detail-row:last-child { border-bottom: 0; }
    .detail-key { color: var(--muted); font-size: 13px; font-weight: 600; }
    .detail-value { color: var(--text); font-size: 13px; font-weight: 700; word-break: break-word; }
    .detail-confidence { color: var(--muted); font-size: 12px; font-weight: 500; margin-top: 2px; }

    .category-alert { margin-top: 12px; padding: 11px 13px; background: var(--amber-bg); border-left: 4px solid var(--review); color: #7A2E0E; font-size: 13px; }
    .why-box { background: #FFFFFF; border: 1px solid var(--border); border-left: 4px solid var(--info); padding: 17px 18px; margin: 20px 0; }
    .why-box h3 { color: var(--navy); margin: 0 0 8px; font-size: 18px; }
    .why-box p { color: #475467; margin: 4px 0; font-size: 14px; line-height: 1.55; }

    div[data-testid="stExpander"] { background: #FFFFFF; border: 1px solid var(--border); border-radius: 0; }
    div[data-testid="stExpander"] details summary { color: var(--navy); font-weight: 800; }

    .technical-table { width: 100%; border-collapse: collapse; background: #FFFFFF; font-size: 13px; }
    .technical-table th { background: #E8EEF7; color: var(--navy); text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); }
    .technical-table td { color: var(--text); padding: 10px 12px; border-bottom: 1px solid #E5EAF2; }
    .technical-table tr:nth-child(even) td { background: #F8FAFC; }

    .summary-box { margin-top: 16px; }
    .footer-note { color: var(--muted); font-size: 12px; line-height: 1.55; border-top: 1px solid var(--border); padding-top: 14px; margin-top: 24px; }

    @media (max-width: 900px) {
        .navbar { align-items: flex-start; flex-direction: column; padding: 12px 18px; }
        .nav-flow { gap: 12px; }
        .nav-meta { white-space: normal; }
        .result-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .decision-strip { grid-template-columns: 1fr; }
        .decision-cell { border-right: 0; border-bottom: 1px solid var(--border); }
        .decision-cell:last-child { border-bottom: 0; }
    }

    @media (max-width: 560px) {
        .result-grid { grid-template-columns: 1fr; }
        .intro h1 { font-size: 26px; }
        .detail-row { grid-template-columns: 1fr; gap: 3px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_all_artifacts():
    binary_model, binary_vectorizer = load_binary_model_artifacts()
    category_artifacts = load_category_model_artifacts()
    rule_engine = RuleEngine()
    return binary_model, binary_vectorizer, category_artifacts, rule_engine


try:
    binary_model, binary_vectorizer, category_artifacts, rule_engine = load_all_artifacts()
except Exception as load_error:
    st.error(
        "The required binary model could not be loaded. Run the training scripts "
        "and verify the files in models/."
    )
    st.exception(load_error)
    st.stop()


# -----------------------------------------------------------------------------
# Display and decision helpers
# -----------------------------------------------------------------------------


def safe_text(value: Any, fallback: str = "Not available") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return fallback
    return text


def pct(value: Any, decimals: int = 1) -> str:
    try:
        if value is None or pd.isna(value):
            return "Not available"
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "Not available"


def confidence_band(value: Any) -> str:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return "Model confidence unavailable"
    if confidence >= 0.80:
        return "Strong model signal"
    if confidence >= 0.70:
        return "Moderate model signal"
    return "Low confidence — review required"


def friendly_trigger(result: Dict[str, Any]) -> str:
    curated = {
        "CURATED_BIRD_STRIKE": "bird strike",
        "CURATED_SMOKE": "smoke or fumes",
        "CURATED_FIRE": "fire",
        "CURATED_ENGINE_SHUTDOWN": "engine shutdown or failure",
        "CURATED_REJECTED_TAKEOFF": "rejected takeoff",
        "CURATED_RUNWAY_INCURSION": "runway incursion",
        "CURATED_HARD_LANDING": "hard landing",
        "CURATED_BALKED_LANDING": "balked landing",
        "CURATED_TCAS_RA": "TCAS / ACAS resolution advisory",
        "CURATED_WINDSHEAR": "windshear",
        "CURATED_FUEL_LEAK": "fuel leak",
        "CURATED_HYDRAULIC_FAILURE": "hydraulic failure or leak",
        "CURATED_GROUND_COLLISION": "ground collision",
        "CURATED_DANGEROUS_GOODS": "dangerous goods",
    }
    primary_id = safe_text(result.get("primary_rule_id"), "")
    if primary_id in curated:
        return curated[primary_id]

    raw = safe_text(result.get("matched_keywords"), "No trigger recorded")
    first = raw.split(";")[0].strip()
    first = first.replace("\\b", "").replace("\\s*", " ").replace("\\s+", " ")
    first = re.sub(r"[\\^$?*+()\[\]{}|]", "", first)
    first = re.sub(r"\s+", " ", first).strip()
    return first or "No trigger recorded"


def category_review(result: Dict[str, Any]) -> Dict[str, str]:
    is_mor = result.get("predicted_mor_label") == 1 or result.get("rule_based_mor") == 1
    if not is_mor:
        return {
            "status": "Not applicable",
            "reason": "Category prediction is not required for this confidently Non-MOR report.",
        }

    checks = [
        ("broad category", result.get("category_level1_confidence")),
        ("event type", result.get("event_type_confidence")),
        ("report title", result.get("report_title_confidence")),
    ]
    low = []
    for label, value in checks:
        try:
            if value is not None and float(value) < 0.60:
                low.append(label)
        except (TypeError, ValueError):
            continue

    if low:
        return {
            "status": "Review recommended",
            "reason": "Low confidence for: " + ", ".join(low) + ".",
        }
    return {"status": "No category review flag", "reason": "All available category confidences are at least 60%."}


def final_review_copy(result: Dict[str, Any]) -> Dict[str, str]:
    required = safe_text(result.get("final_review_required"), "No")
    priority = safe_text(result.get("final_review_priority"), "None")
    reason = safe_text(result.get("final_review_reason"), "No configured review condition was triggered.")

    if required == "No":
        return {
            "css": "",
            "title": "No Automated Review Flag",
            "reason": "No configured review condition was triggered. Final human oversight remains applicable.",
        }
    if priority == "Mandatory":
        return {
            "css": "mandatory",
            "title": "Mandatory Review",
            "reason": "The rule engine identified an MOR condition while the ML model predicted Non-MOR.",
        }
    if priority == "Required":
        return {
            "css": "required",
            "title": "Human Review Required",
            "reason": reason or "The model confidence is below the configured acceptance threshold.",
        }
    return {
        "css": "recommended",
        "title": "Review Recommended",
        "reason": reason or "The model predicted Non-MOR, but the confidence is below 80%.",
    }


def html_detail_row(key: str, value: Any, confidence: Any = None) -> str:
    display = html.escape(safe_text(value))
    conf = ""
    if confidence is not None:
        conf = f'<div class="detail-confidence">Confidence: {html.escape(pct(confidence))}</div>'
    return (
        '<div class="detail-row">'
        f'<div class="detail-key">{html.escape(key)}</div>'
        f'<div class="detail-value">{display}{conf}</div>'
        '</div>'
    )


def run_unified_prediction(report_text: str) -> Dict[str, Any]:
    """Call the unified predictor while supporting minor signature differences."""
    signature = inspect.signature(predict_single_report)
    kwargs: Dict[str, Any] = {"report_text": report_text}
    if "binary_model" in signature.parameters:
        kwargs["binary_model"] = binary_model
    elif "model" in signature.parameters:
        kwargs["model"] = binary_model
    if "binary_vectorizer" in signature.parameters:
        kwargs["binary_vectorizer"] = binary_vectorizer
    elif "vectorizer" in signature.parameters:
        kwargs["vectorizer"] = binary_vectorizer
    kwargs["rule_engine"] = rule_engine
    if "category_artifacts" in signature.parameters:
        kwargs["category_artifacts"] = category_artifacts
    return predict_single_report(**kwargs)


def build_summary(result: Dict[str, Any], review: Dict[str, str]) -> str:
    rule_label = safe_text(result.get("primary_subcategory"), "No MOR rule matched")
    category = safe_text(result.get("predicted_category_level1"), "Not applicable")
    event_type = safe_text(result.get("predicted_event_type"), "Not applicable")
    return "\n".join(
        [
            f"MOR Classification: {safe_text(result.get('predicted_label_text'))}",
            f"Confidence: {pct(result.get('prediction_confidence'))}",
            f"Rule Match: {rule_label}",
            f"Category: {category}",
            f"Event Type: {event_type}",
            f"Review Requirement: {review['title']}",
        ]
    )


# -----------------------------------------------------------------------------
# Navigation and introduction
# -----------------------------------------------------------------------------

category_count = len(category_artifacts) if isinstance(category_artifacts, dict) else 0
system_limited = category_count < 3
system_text = (
    f"System available with limited category prediction · {category_count}/3 category models loaded"
    if system_limited
    else "System ready · Binary model loaded · 3 category models loaded · Rules loaded"
)

st.markdown(
    f"""
    <div class="navbar-shell">
      <div class="navbar">
        <div class="brand-area">
          <div class="brand-mark">6E</div>
          <div>
            <div class="brand-title">Flight Safety MOR Classifier</div>
            <div class="brand-subtitle">Internal decision-support prototype</div>
          </div>
        </div>
        <div class="nav-flow" aria-label="Application workflow">
          <span>Report Input</span><span>Classification</span><span>Rule Review</span><span>Category</span><span>Final Decision</span>
        </div>
        <div class="nav-meta">Model v{html.escape(MODEL_VERSION)}</div>
      </div>
    </div>
    <div class="intro">
      <h1>Mandatory Occurrence Report Classification Assistant</h1>
      <p>Paste a safety-report narrative to generate an MOR/Non-MOR recommendation, confidence score, rule match, likely category, and human-review requirement.</p>
      <p class="intro-note">This output supports Flight Safety review and does not replace the final reporting decision.</p>
    </div>
    <div class="status-line"><span class="status-dot"></span>{html.escape(system_text)}</div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Input workflow
# -----------------------------------------------------------------------------

st.markdown('<div class="section-title">Report Input</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">Enter the complete operational narrative. The report is processed locally by the host system.</div>',
    unsafe_allow_html=True,
)

report_text = st.text_area(
    "Paste report text",
    key="report_text",
    height=245,
    placeholder=(
        "Example: During initial climb, the aircraft experienced a suspected bird strike. "
        "Aircraft parameters remained normal and engineering inspection was requested after landing."
    ),
    label_visibility="collapsed",
)

trimmed = " ".join(report_text.split())
word_count = len(trimmed.split()) if trimmed else 0
character_count = len(report_text)
st.markdown(
    f'<div class="count-line">{word_count} words · {character_count} characters</div>',
    unsafe_allow_html=True,
)

button_cols = st.columns([1.25, 1, 1, 3.5])
with button_cols[0]:
    predict_clicked = st.button("Predict Report", use_container_width=True)
with button_cols[1]:
    st.button("Load Example", on_click=load_example, use_container_width=True)
with button_cols[2]:
    st.button("Clear Input", on_click=clear_report, use_container_width=True)

st.markdown(
    '<div class="privacy-note"><strong>Internal use only.</strong> Do not paste operational safety data into a publicly hosted deployment. When using a Network URL, access must remain within an authorized corporate network; processing occurs on the host system, and public exposure must not be enabled.</div>',
    unsafe_allow_html=True,
)

if predict_clicked:
    if not trimmed:
        st.session_state.input_message = "Please enter a report before running classification."
    elif word_count < 5:
        st.session_state.input_message = "The report is too short. Enter at least five meaningful words for a useful classification."
    else:
        st.session_state.input_message = ""
        if word_count > 1500:
            st.warning("The narrative is unusually long. The report will be processed, but consider removing unrelated material.")

        with st.status("Analyzing report…", expanded=True) as process_status:
            st.write("Running binary classifier")
            st.write("Checking MOR rules")
            st.write("Predicting category")
            st.write("Applying review logic")
            try:
                result = run_unified_prediction(trimmed)
                st.session_state.prediction_result = result
                st.session_state.prediction_completed = True
                process_status.update(label="Analysis complete", state="complete", expanded=False)
            except Exception as prediction_error:
                st.session_state.prediction_completed = False
                process_status.update(label="Analysis could not be completed", state="error", expanded=True)
                st.exception(prediction_error)

if st.session_state.input_message:
    st.warning(st.session_state.input_message)


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------

if st.session_state.prediction_completed and st.session_state.prediction_result:
    result = st.session_state.prediction_result
    review = final_review_copy(result)
    category_status = category_review(result)

    ml_label = safe_text(result.get("predicted_label_text"))
    rule_matched = result.get("rule_based_mor") == 1
    confidence = result.get("prediction_confidence")
    confidence_width = max(0.0, min(100.0, float(confidence or 0) * 100))
    review_label = review["title"]

    ml_class = "critical" if ml_label == "MOR" else "safe"
    ml_text = "critical-text" if ml_label == "MOR" else "safe-text"
    rule_class = "critical" if rule_matched else "safe"
    rule_text_class = "critical-text" if rule_matched else "safe-text"
    review_class = "critical" if review["css"] == "mandatory" else ("review" if review["css"] else "safe")
    review_text_class = "critical-text" if review["css"] == "mandatory" else ("review-text" if review["css"] else "safe-text")

    st.markdown("---")
    st.markdown('<div class="section-title">Classification Output</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="result-grid">
          <div class="result-card {ml_class}">
            <div class="card-label">MOR Classification</div>
            <div class="card-value {ml_text}">{html.escape(ml_label)}</div>
            <div class="card-note">Machine-learning prediction</div>
          </div>
          <div class="result-card">
            <div class="card-label">Prediction Confidence</div>
            <div class="card-value">{html.escape(pct(confidence))}</div>
            <div class="confidence-track"><div class="confidence-fill" style="width:{confidence_width:.1f}%"></div></div>
            <div class="card-note">{html.escape(confidence_band(confidence))}</div>
          </div>
          <div class="result-card {rule_class}">
            <div class="card-label">Rule Status</div>
            <div class="card-value {rule_text_class}">{'Rule Matched' if rule_matched else 'No Match'}</div>
            <div class="card-note">{'MOR trigger identified' if rule_matched else 'No configured trigger found'}</div>
          </div>
          <div class="result-card {review_class}">
            <div class="card-label">Review Requirement</div>
            <div class="card-value {review_text_class}">{html.escape(review_label)}</div>
            <div class="card-note">Automated workflow flag</div>
          </div>
        </div>
        <div class="review-banner {html.escape(review['css'])}">
          <div class="review-title">{html.escape(review['title'])}</div>
          <div class="review-reason">{html.escape(review['reason'])}</div>
        </div>
        <div class="decision-strip">
          <div class="decision-cell"><div class="decision-label">ML Prediction</div><div class="decision-value">{html.escape(ml_label)}</div></div>
          <div class="decision-cell"><div class="decision-label">Rule Engine Result</div><div class="decision-value">{'MOR trigger matched' if rule_matched else 'No MOR trigger matched'}</div></div>
          <div class="decision-cell"><div class="decision-label">Final Recommendation</div><div class="decision-value">{html.escape(safe_text(result.get('final_recommendation'), review['title']))}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        rule_rows = "".join(
            [
                html_detail_row("Rule Match", "Yes" if rule_matched else "No"),
                html_detail_row("Primary Rule", result.get("primary_rule_id"), result.get("rule_confidence")),
                html_detail_row("Matched Trigger", friendly_trigger(result)),
                html_detail_row("Rule Domain", result.get("primary_domain")),
                html_detail_row("Rule Sub-category", result.get("primary_subcategory")),
                html_detail_row("Rules Matched", result.get("matched_rule_count", 0)),
            ]
        )
        st.markdown(
            f'<div class="detail-panel"><div class="detail-heading">Rule Match Details</div>{rule_rows}</div>',
            unsafe_allow_html=True,
        )

        matched_count = int(result.get("matched_rule_count") or 0)
        if matched_count > 1:
            with st.expander(f"Secondary rule matches ({matched_count - 1})"):
                st.write("Matched rule IDs:", safe_text(result.get("matched_rule_ids")))
                st.write("Matched sub-categories:", safe_text(result.get("matched_subcategories")))

    with right:
        category_attempted = any(
            result.get(key) not in {None, "", "None"}
            for key in ["predicted_category_level1", "predicted_event_type", "predicted_report_title"]
        )
        if category_attempted:
            category_rows = "".join(
                [
                    html_detail_row("Broad Category", result.get("predicted_category_level1"), result.get("category_level1_confidence")),
                    html_detail_row("Event Type", result.get("predicted_event_type"), result.get("event_type_confidence")),
                    html_detail_row("Report Title / Sub-category", result.get("predicted_report_title"), result.get("report_title_confidence")),
                    html_detail_row("Category Status", category_status["status"]),
                ]
            )
            category_body = category_rows
        else:
            category_body = (
                '<div class="section-copy">Category prediction is not applicable for this report, '
                'or the required category model is unavailable.</div>'
            )
        st.markdown(
            f'<div class="detail-panel"><div class="detail-heading">Category / Sub-category</div>{category_body}</div>',
            unsafe_allow_html=True,
        )
        if category_status["status"] == "Review recommended":
            st.markdown(
                f'<div class="category-alert"><strong>Category Review Recommended</strong><br>{html.escape(category_status["reason"])}</div>',
                unsafe_allow_html=True,
            )

    mor_probability = pct(result.get("mor_probability"))
    event_category = safe_text(result.get("predicted_category_level1"), "not available")
    why_lines = [f"The ML classifier assigned a {mor_probability} probability to MOR."]
    if rule_matched:
        why_lines.append(f'The rule engine matched the “{friendly_trigger(result)}” condition.')
    else:
        why_lines.append("The rule engine did not identify a configured MOR trigger.")
    if category_attempted:
        why_lines.append(f"The predicted broad event category is {event_category}.")

    st.markdown(
        '<div class="why-box"><h3>Why this result?</h3>'
        + "".join(f"<p>{html.escape(line)}</p>" for line in why_lines)
        + "</div>",
        unsafe_allow_html=True,
    )

    technical_rows = [
        ("MOR probability", pct(result.get("mor_probability"), 2)),
        ("Non-MOR probability", pct(result.get("non_mor_probability"), 2)),
        ("Classification threshold", f"{MOR_THRESHOLD:.2f}"),
        ("Predicted-class confidence", pct(result.get("prediction_confidence"), 2)),
        ("Rule-based MOR flag", "1" if rule_matched else "0"),
        ("Rule confidence", pct(result.get("rule_confidence"), 2)),
        ("Primary rule ID", safe_text(result.get("primary_rule_id"))),
        ("Broad-category confidence", pct(result.get("category_level1_confidence"), 2)),
        ("Event-type confidence", pct(result.get("event_type_confidence"), 2)),
        ("Report-title confidence", pct(result.get("report_title_confidence"), 2)),
        ("Binary model version", MODEL_VERSION),
        ("Category model version", safe_text(MODEL_METADATA.get("category_model_version"))),
    ]

    with st.expander("Technical Details"):
        rows_html = "".join(
            f"<tr><td>{html.escape(field)}</td><td>{html.escape(str(value))}</td></tr>"
            for field, value in technical_rows
        )
        st.markdown(
            f'<table class="technical-table"><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>{rows_html}</tbody></table>',
            unsafe_allow_html=True,
        )
        with st.expander("Developer rule data"):
            st.write("Matched rule IDs:", safe_text(result.get("matched_rule_ids")))
            st.write("Raw matched keywords:", safe_text(result.get("matched_keywords")))
            st.write("Primary condition:", safe_text(result.get("primary_condition")))

    summary = build_summary(result, review)
    st.markdown('<div class="section-title summary-box">Copyable Summary</div>', unsafe_allow_html=True)
    st.code(summary, language=None)
    st.download_button(
        "Download Summary",
        data=summary,
        file_name="mor_classification_summary.txt",
        mime="text/plain",
    )

    st.button("Clear and Analyze Another Report", on_click=clear_report)

st.markdown(
    """
    <div class="footer-note">
      <strong>Confidentiality:</strong> This local prototype is intended for authorized internal use. Report narratives may contain confidential operational information. Do not expose the application through public hosting, public Wi-Fi, port forwarding, or an unrestricted Network URL. Final MOR and regulatory reporting decisions remain with authorized Flight Safety personnel.
    </div>
    """,
    unsafe_allow_html=True,
)
