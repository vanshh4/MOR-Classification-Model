
"""
streamlit_app.py

Purpose:
    Streamlit UI for the MOR Classification Model used by Flight Safety.

Features:
    1. Paste report text
    2. Predict MOR / Non-MOR
    3. Show confidence
    4. Show rule match
    5. Show predicted category/sub-category
    6. Show final review requirement

Expected project structure:
    MOR_Classification_Project/
    ├── app/
    │   └── streamlit_app.py
    ├── src/
    │   ├── predict_mor_with_rules.py
    │   └── rule_engine.py
    ├── models/
    │   ├── mor_binary_model.pkl
    │   ├── mor_binary_vectorizer.pkl
    │   ├── mor_category_level1_model.pkl
    │   ├── mor_category_level1_vectorizer.pkl
    │   ├── mor_event_type_model.pkl
    │   ├── mor_event_type_vectorizer.pkl
    │   ├── mor_report_title_model.pkl
    │   └── mor_report_title_vectorizer.pkl

Run from project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT PATH SETUP
# ============================================================


def find_project_root() -> Path:
    """Find project root when this file is stored in app/streamlit_app.py."""
    current_file = Path(__file__).resolve()

    if current_file.parent.name.lower() == "app":
        return current_file.parents[1]

    for parent in [current_file.parent] + list(current_file.parents):
        if (parent / "src").exists() and (parent / "models").exists():
            return parent

    return Path.cwd()


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


# Import unified prediction logic from src/predict_mor_with_rules.py
try:
    from predict_mor_with_rules import (  # type: ignore
        load_binary_model_artifacts,
        load_category_model_artifacts,
        predict_single_report,
    )
    from rule_engine import RuleEngine  # type: ignore
except Exception as import_error:
    st.error("Unable to import model prediction modules. Check that src/ files exist.")
    st.exception(import_error)
    st.stop()


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IndiGo MOR Classification",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 3. CUSTOM CSS - MODERN INDIGO FLIGHT SAFETY DESIGN
# ============================================================

st.markdown(
    """
    <style>
        /* Main app background */
        .stApp {
            background: linear-gradient(135deg, #f6f9ff 0%, #edf3ff 45%, #ffffff 100%);
            color: #111827;
        }

        /* Hide Streamlit default clutter */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Main container spacing */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1300px;
        }

        /* Indigo header */
        .hero-card {
            background: linear-gradient(135deg, #071c4d 0%, #123b8f 50%, #1f6feb 100%);
            padding: 2.2rem 2.4rem;
            border-radius: 24px;
            color: white;
            box-shadow: 0 24px 60px rgba(7, 28, 77, 0.22);
            margin-bottom: 1.5rem;
        }

        .brand-label {
            font-size: 0.85rem;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #bcd7ff;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }

        .hero-title {
            font-size: 2.25rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.65rem;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            line-height: 1.6;
            color: #e8f1ff;
            max-width: 920px;
        }

        /* Cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid rgba(204, 219, 244, 0.9);
            border-radius: 20px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 14px 38px rgba(15, 23, 42, 0.08);
            min-height: 132px;
        }

        .metric-label {
            font-size: 0.78rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }

        .metric-value {
            font-size: 1.7rem;
            line-height: 1.2;
            color: #0f172a;
            font-weight: 800;
        }

        .metric-subtext {
            color: #64748b;
            font-size: 0.88rem;
            margin-top: 0.45rem;
        }

        .status-mor {
            color: #b91c1c;
        }

        .status-nonmor {
            color: #047857;
        }

        .status-review {
            color: #b45309;
        }

        .section-card {
            background: rgba(255, 255, 255, 0.97);
            border: 1px solid rgba(204, 219, 244, 0.9);
            border-radius: 22px;
            padding: 1.5rem 1.6rem;
            box-shadow: 0 16px 42px rgba(15, 23, 42, 0.07);
            margin-top: 1rem;
        }

        .section-title {
            color: #0f2f75;
            font-size: 1.18rem;
            font-weight: 800;
            margin-bottom: 0.6rem;
        }

        .info-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.65rem 0;
            border-bottom: 1px solid #e5edf9;
        }

        .info-row:last-child {
            border-bottom: none;
        }

        .info-key {
            color: #64748b;
            font-weight: 700;
            min-width: 190px;
        }

        .info-value {
            color: #0f172a;
            font-weight: 650;
            text-align: right;
            word-break: break-word;
        }

        .review-banner-green {
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            color: #065f46;
            padding: 1rem 1.2rem;
            border-radius: 18px;
            font-weight: 800;
        }

        .review-banner-amber {
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
            padding: 1rem 1.2rem;
            border-radius: 18px;
            font-weight: 800;
        }

        .review-banner-red {
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
            padding: 1rem 1.2rem;
            border-radius: 18px;
            font-weight: 800;
        }

        /* Streamlit buttons */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #0f3b8f 0%, #1f6feb 100%);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 0.75rem 1.25rem;
            font-weight: 800;
            box-shadow: 0 12px 25px rgba(31,111,235,0.22);
        }

        div.stButton > button:first-child:hover {
            background: linear-gradient(135deg, #092b6b 0%, #155bd4 100%);
            color: white;
            border: none;
        }

        textarea {
            border-radius: 16px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. MODEL LOADING
# ============================================================


@st.cache_resource(show_spinner=False)
def load_all_artifacts():
    """Load binary model, category models, and rule engine once."""
    binary_model, binary_vectorizer = load_binary_model_artifacts()
    category_artifacts = load_category_model_artifacts()
    rule_engine = RuleEngine()
    return binary_model, binary_vectorizer, category_artifacts, rule_engine


try:
    binary_model, binary_vectorizer, category_artifacts, rule_engine = load_all_artifacts()
    artifacts_loaded = True
except Exception as load_error:
    artifacts_loaded = False
    binary_model = binary_vectorizer = category_artifacts = rule_engine = None


# ============================================================
# 5. DISPLAY HELPERS
# ============================================================


def format_probability(value: Any) -> str:
    """Format probability as percentage."""
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "N/A"


def none_to_dash(value: Any) -> str:
    """Display empty values as dash."""
    if value is None:
        return "—"
    if isinstance(value, float) and pd.isna(value):
        return "—"
    text = str(value).strip()
    return text if text else "—"


def render_metric_card(label: str, value: str, subtext: str = "", css_class: str = "") -> None:
    """Render custom metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {css_class}">{value}</div>
            <div class="metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_row(key: str, value: Any) -> None:
    """Render a clean key-value row."""
    st.markdown(
        f"""
        <div class="info-row">
            <div class="info-key">{key}</div>
            <div class="info-value">{none_to_dash(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_review_banner(result: Dict[str, Any]) -> None:
    """Show final review status banner."""
    review_required = result.get("final_review_required", "No")
    priority = result.get("final_review_priority", "None")
    reason = result.get("final_review_reason", "No review rule triggered")

    if review_required == "No":
        css = "review-banner-green"
        title = "No Review Required"
    elif priority == "Mandatory":
        css = "review-banner-red"
        title = "Mandatory Review Required"
    else:
        css = "review-banner-amber"
        title = f"Review {priority}"

    st.markdown(
        f"""
        <div class="{css}">
            {title}<br>
            <span style="font-weight:600;">Reason: {reason}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 6. HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="brand-label">IndiGo Flight Safety · MOR Intelligence</div>
        <div class="hero-title">Mandatory Occurrence Report Classification</div>
        <div class="hero-subtitle">
            Paste a safety report narrative to classify whether it is likely to be a Mandatory Occurrence Report.
            The system combines an ML classifier, rule-based MOR checks, and category prediction to support
            Flight Safety review decisions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 7. MODEL STATUS
# ============================================================

if not artifacts_loaded:
    st.error("Model artifacts could not be loaded. Please ensure model files exist in the models/ folder.")
    st.exception(load_error)
    st.stop()


# ============================================================
# 8. INPUT AREA
# ============================================================

left_col, right_col = st.columns([1.25, 0.75], gap="large")

with left_col:
    st.markdown("### Report Input")
    report_text = st.text_area(
        label="Paste report text",
        placeholder=(
            "Example: Aircraft experienced bird strike during climb after departure. "
            "Engineering inspection was carried out after landing."
        ),
        height=230,
        label_visibility="collapsed",
    )

    button_col_1, button_col_2 = st.columns([0.32, 0.68])
    with button_col_1:
        predict_clicked = st.button("Predict MOR", use_container_width=True)
    with button_col_2:
        st.caption("The prediction is advisory and should be reviewed by Flight Safety for final reporting decisions.")

with right_col:
    st.markdown("### System Status")
    category_count = len(category_artifacts) if isinstance(category_artifacts, dict) else 0

    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">Loaded Components</div>
            <div class="info-row">
                <div class="info-key">Binary MOR Model</div>
                <div class="info-value">Loaded</div>
            </div>
            <div class="info-row">
                <div class="info-key">Rule Engine</div>
                <div class="info-value">Loaded</div>
            </div>
            <div class="info-row">
                <div class="info-key">Category Models</div>
                <div class="info-value">{category_count}/3 Loaded</div>
            </div>
            <div class="info-row">
                <div class="info-key">Mode</div>
                <div class="info-value">Single Report Prediction</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 9. PREDICTION EXECUTION
# ============================================================

if predict_clicked:
    if not report_text or not report_text.strip():
        st.warning("Please paste a report text before running prediction.")
        st.stop()

    with st.spinner("Analyzing report using ML model and MOR rule engine..."):
        result = predict_single_report(
            report_text=report_text,
            binary_model=binary_model,
            binary_vectorizer=binary_vectorizer,
            rule_engine=rule_engine,
            category_artifacts=category_artifacts,
        )

    st.markdown("---")
    st.markdown("## Prediction Result")

    # Main summary cards
    pred_label = result.get("predicted_label_text", "N/A")
    pred_class = "status-mor" if pred_label == "MOR" else "status-nonmor"

    review_required = result.get("final_review_required", "N/A")
    review_priority = result.get("final_review_priority", "None")
    review_class = "status-review" if review_required == "Yes" else "status-nonmor"

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
        render_metric_card(
            label="ML Prediction",
            value=pred_label,
            subtext="Binary MOR classifier output",
            css_class=pred_class,
        )

    with c2:
        render_metric_card(
            label="Prediction Confidence",
            value=format_probability(result.get("prediction_confidence")),
            subtext="Probability of predicted ML label",
        )

    with c3:
        rule_text = "Matched" if result.get("rule_based_mor") == 1 else "No Match"
        rule_class = "status-mor" if rule_text == "Matched" else "status-nonmor"
        render_metric_card(
            label="Rule Engine",
            value=rule_text,
            subtext=f"Rule confidence: {format_probability(result.get('rule_confidence'))}",
            css_class=rule_class,
        )

    with c4:
        render_metric_card(
            label="Review Status",
            value=review_priority if review_required == "Yes" else "Not Required",
            subtext="Final review decision",
            css_class=review_class,
        )

    st.markdown("### Final Review Requirement")
    render_review_banner(result)

    # Detailed output sections
    detail_col_1, detail_col_2 = st.columns(2, gap="large")

    with detail_col_1:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Rule Match Details</div>
            """,
            unsafe_allow_html=True,
        )
        render_info_row("Rule-based MOR", "Yes" if result.get("rule_based_mor") == 1 else "No")
        render_info_row("Matched Rule Count", result.get("matched_rule_count"))
        render_info_row("Matched Rule IDs", result.get("matched_rule_ids"))
        render_info_row("Matched Keywords", result.get("matched_keywords"))
        render_info_row("Primary Subcategory", result.get("primary_subcategory"))
        render_info_row("Primary Condition", result.get("primary_condition"))
        st.markdown("</div>", unsafe_allow_html=True)

    with detail_col_2:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Predicted Category / Sub-category</div>
            """,
            unsafe_allow_html=True,
        )
        render_info_row("Broad Category", result.get("predicted_category_level1"))
        render_info_row("Category Confidence", format_probability(result.get("category_level1_confidence")))
        render_info_row("Event Type", result.get("predicted_event_type"))
        render_info_row("Event Type Confidence", format_probability(result.get("event_type_confidence")))
        render_info_row("Report Title / Sub-category", result.get("predicted_report_title"))
        render_info_row("Sub-category Confidence", format_probability(result.get("report_title_confidence")))
        st.markdown("</div>", unsafe_allow_html=True)

    # Optional expandable technical details
    with st.expander("Show technical prediction details"):
        technical_df = pd.DataFrame(
            [
                {"Field": "MOR Probability", "Value": format_probability(result.get("mor_probability"))},
                {"Field": "Non-MOR Probability", "Value": format_probability(result.get("non_mor_probability"))},
                {"Field": "Final Recommendation", "Value": none_to_dash(result.get("final_recommendation"))},
                {"Field": "Final Review Required", "Value": none_to_dash(result.get("final_review_required"))},
                {"Field": "Final Review Priority", "Value": none_to_dash(result.get("final_review_priority"))},
                {"Field": "Final Review Reason", "Value": none_to_dash(result.get("final_review_reason"))},
                {"Field": "Primary Rule ID", "Value": none_to_dash(result.get("primary_rule_id"))},
                {"Field": "Primary Domain", "Value": none_to_dash(result.get("primary_domain"))},
            ]
        )
        st.dataframe(technical_df, use_container_width=True, hide_index=True)

else:
    st.markdown("---")
    st.info("Paste a safety report and click **Predict MOR** to view the classification result.")
