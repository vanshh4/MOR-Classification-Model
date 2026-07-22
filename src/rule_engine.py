"""
rule_engine.py

Purpose:
    Rule-based MOR detection layer for the MOR Classification project.

What it does:
    - Loads structured MOR rules from:
        data/processed/rules/structured_MOR_rules.xlsx
    - Applies keyword/phrase-based MOR checks on report text.
    - Includes a curated set of high-priority MOR trigger patterns such as:
        bird strike, smoke, fire, engine shutdown, rejected takeoff,
        runway incursion, hard landing, balked landing, TCAS RA, windshear,
        fuel leak, hydraulic failure, ground collision, dangerous goods.
    - Returns matched rule IDs, categories, subcategories, matched keywords,
      and rule_based_mor flag.

Expected usage:
    from rule_engine import RuleEngine

    engine = RuleEngine()
    result = engine.evaluate_report("Aircraft experienced bird strike during climb.")
    print(result)

Standalone usage:
    python src/rule_engine.py

Integration logic with ML model:
    If rule_engine says MOR and ML model says Non-MOR:
        Mandatory Review
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ============================================================
# 1. PROJECT PATH CONFIGURATION
# ============================================================


def find_project_root() -> Path:
    """
    Find project root robustly.

    Expected location:
        MOR_Classification_Project/src/rule_engine.py

    If the script is run directly from another folder, it falls back to cwd.
    """
    script_path = Path(__file__).resolve()

    if script_path.parent.name.lower() == "src":
        return script_path.parents[1]

    for parent in [script_path.parent] + list(script_path.parents):
        if (parent / "data" / "processed").exists():
            return parent

    return Path.cwd()


PROJECT_ROOT = find_project_root()

DEFAULT_RULES_FILE = PROJECT_ROOT / "data" / "processed" / "rules" / "structured_MOR_rules.xlsx"
FALLBACK_RULES_FILE = Path("structured_MOR_rules.xlsx")


# ============================================================
# 2. DATA CLASSES
# ============================================================


@dataclass
class RuleMatch:
    """Represents one matched rule/pattern."""

    rule_id: str
    source: str
    domain: str
    subcategory: str
    mor_condition: str
    matched_keyword: str
    match_type: str
    confidence: float


@dataclass
class RuleEngineResult:
    """Final rule engine output for one report."""

    rule_based_mor: int
    rule_confidence: float
    matched_rule_count: int
    matched_rule_ids: str
    matched_keywords: str
    matched_domains: str
    matched_subcategories: str
    primary_rule_id: Optional[str]
    primary_domain: Optional[str]
    primary_subcategory: Optional[str]
    primary_condition: Optional[str]
    review_hint: str
    matches: List[Dict]


# ============================================================
# 3. CURATED HIGH-PRIORITY MOR TRIGGER PATTERNS
# ============================================================

# These curated patterns are intentionally included even if structured_MOR_rules.xlsx
# is unavailable. They cover high-priority trigger phrases requested for this project.
#
# Pattern design notes:
# - Each pattern uses lowercase text.
# - Regex boundaries are used where helpful.
# - The engine applies these patterns after normalizing the incoming report text.

CURATED_MOR_PATTERNS: List[Dict] = [
    {
        "rule_id": "CURATED_BIRD_STRIKE",
        "domain": "Wildlife Activity / Flight Operations",
        "subcategory": "Bird/Wildlife Strike",
        "mor_condition": "Bird strike or wildlife strike reported or suspected.",
        "patterns": [
            r"\bbird\s*strike\b",
            r"\bbirdstrike\b",
            r"\bbird\s*hit\b",
            r"\bwildlife\s*strike\b",
            r"\bbird\s*impact\b",
            r"\bbird\s*ingestion\b",
        ],
        "confidence": 0.95,
    },
    {
        "rule_id": "CURATED_SMOKE",
        "domain": "Aircraft Flight Operations / Emergencies",
        "subcategory": "Smoke/Fumes",
        "mor_condition": "Smoke, fumes, toxic smell, or smoke warning reported.",
        "patterns": [
            r"\bsmoke\b",
            r"\bsmoke\s*warning\b",
            r"\bsmoke\s*ecam\b",
            r"\bfumes?\b",
            r"\bburning\s*smell\b",
            r"\btoxic\s*fumes?\b",
        ],
        "confidence": 0.90,
    },
    {
        "rule_id": "CURATED_FIRE",
        "domain": "Aircraft Flight Operations / Emergencies",
        "subcategory": "Fire/Explosion",
        "mor_condition": "Fire, explosion, or fire warning reported.",
        "patterns": [
            r"\bfire\b",
            r"\bfire\s*warning\b",
            r"\bengine\s*fire\b",
            r"\bbrake\s*fire\b",
            r"\bexplosion\b",
        ],
        "confidence": 0.93,
    },
    {
        "rule_id": "CURATED_ENGINE_SHUTDOWN",
        "domain": "Aircraft Technical",
        "subcategory": "Propulsion System",
        "mor_condition": "Engine shutdown, flameout, or engine failure/malfunction reported.",
        "patterns": [
            r"\bengine\s*shutdown\b",
            r"\bengine\s*shut\s*down\b",
            r"\bin[-\s]*flight\s*engine\s*shutdown\b",
            r"\bengine\s*failure\b",
            r"\bflameout\b",
            r"\bengine\s*malfunction\b",
        ],
        "confidence": 0.95,
    },
    {
        "rule_id": "CURATED_REJECTED_TAKEOFF",
        "domain": "Aircraft Flight Operations",
        "subcategory": "Rejected Takeoff",
        "mor_condition": "Rejected takeoff, aborted takeoff, or RTO reported.",
        "patterns": [
            r"\brejected\s*take[-\s]*off\b",
            r"\brejected\s*takeoff\b",
            r"\baborted\s*take[-\s]*off\b",
            r"\bRTO\b",
            r"\breject\s*the\s*take[-\s]*off\b",
        ],
        "confidence": 0.95,
    },
    {
        "rule_id": "CURATED_RUNWAY_INCURSION",
        "domain": "Aircraft Flight Operations / Air Navigation Services",
        "subcategory": "Runway Incursion",
        "mor_condition": "Runway incursion, unauthorized runway entry, or occupied runway event reported.",
        "patterns": [
            r"\brunway\s*incursion\b",
            r"\bentered\s*runway\b",
            r"\bunauthori[sz]ed\s*runway\b",
            r"\boccupied\s*runway\b",
            r"\bclosed\s*runway\b",
            r"\bincorrect\s*runway\b",
        ],
        "confidence": 0.95,
    },
    {
        "rule_id": "CURATED_HARD_LANDING",
        "domain": "Aircraft Flight Operations",
        "subcategory": "Hard Landing / High G Landing",
        "mor_condition": "Hard landing or high-G landing reported or suspected.",
        "patterns": [
            r"\bhard\s*landing\b",
            r"\bsuspected\s*hard\s*landing\b",
            r"\bhigh\s*g\s*landing\b",
            r"\bhigh-g\s*landing\b",
            r"\bfirm\s*touchdown\b",
            r"\bVRTA\b",
        ],
        "confidence": 0.88,
    },
    {
        "rule_id": "CURATED_BALKED_LANDING",
        "domain": "Aircraft Flight Operations",
        "subcategory": "Balked Landing / Go Around Below Minima",
        "mor_condition": "Balked landing or go-around below minima reported.",
        "patterns": [
            r"\bbalked\s*landing\b",
            r"\bgo[-\s]*around\s*below\s*minima\b",
            r"\bgo\s*around\s*below\s*minimums\b",
            r"\blow\s*level\s*go[-\s]*around\b",
            r"\bwheels?\s*.*\bcontact\b.*\bgo[-\s]*around\b",
            r"\btouch\s*and\s*go[-\s]*around\b",
        ],
        "confidence": 0.90,
    },
    {
        "rule_id": "CURATED_TCAS_RA",
        "domain": "Aircraft Flight Operations",
        "subcategory": "TCAS/ACAS Resolution Advisory",
        "mor_condition": "TCAS RA or ACAS RA reported.",
        "patterns": [
            r"\bTCAS\s*RA\b",
            r"\bACAS\s*RA\b",
            r"\bresolution\s*advisory\b",
            r"\btraffic\s*resolution\s*advisory\b",
        ],
        "confidence": 0.95,
    },
    {
        "rule_id": "CURATED_WINDSHEAR",
        "domain": "Aircraft Flight Operations / Meteorology",
        "subcategory": "Windshear Encounter",
        "mor_condition": "Windshear warning, alert, or encounter reported.",
        "patterns": [
            r"\bwindshear\b",
            r"\bwind\s*shear\b",
            r"\bwindshear\s*warning\b",
            r"\bwindshear\s*alert\b",
            r"\bwind\s*shear\s*encounter\b",
        ],
        "confidence": 0.93,
    },
    {
        "rule_id": "CURATED_FUEL_LEAK",
        "domain": "Aircraft Technical",
        "subcategory": "Fuel System",
        "mor_condition": "Fuel leak, fuel leakage, or significant fuel system leak reported.",
        "patterns": [
            r"\bfuel\s*leak\b",
            r"\bfuel\s*leakage\b",
            r"\bleakage\s*of\s*fuel\b",
            r"\bfuel\s*spill\b",
            r"\bfuel\s*spillage\b",
        ],
        "confidence": 0.92,
    },
    {
        "rule_id": "CURATED_HYDRAULIC_FAILURE",
        "domain": "Aircraft Technical",
        "subcategory": "Hydraulics",
        "mor_condition": "Hydraulic system failure, loss, or hydraulic fluid leakage reported.",
        "patterns": [
            r"\bhydraulic\s*failure\b",
            r"\bhydraulic\s*system\s*loss\b",
            r"\bloss\s*of\s*hydraulic\b",
            r"\bhydraulic\s*leak\b",
            r"\bhydraulic\s*fluid\s*leak\b",
        ],
        "confidence": 0.92,
    },
    {
        "rule_id": "CURATED_GROUND_COLLISION",
        "domain": "Ground Operations / Aircraft Flight Operations",
        "subcategory": "Ground Collision / Ground Damage",
        "mor_condition": "Collision with aircraft, vehicle, ground equipment, or ground object reported.",
        "patterns": [
            r"\bground\s*collision\b",
            r"\bcollision\s*with\s*(?:vehicle|aircraft|equipment|object)\b",
            r"\baircraft\s*.*\bcontacted\s*.*\b(vehicle|equipment|object|gpu|apu|trolley|cart)\b",
            r"\b(vehicle|equipment|gpu|trolley|cart)\s*.*\bcontacted\s*.*\baircraft\b",
            r"\bwingtip\s*.*\bcontact\b",
            r"\btail\s*.*\bcontact\b",
            r"\bground\s*damage\b",
        ],
        "confidence": 0.90,
    },
    {
        "rule_id": "CURATED_DANGEROUS_GOODS",
        "domain": "Passenger Handling, Baggage and Cargo",
        "subcategory": "Dangerous Goods",
        "mor_condition": "Dangerous goods incident, undeclared DG, or misdeclared DG reported.",
        "patterns": [
            r"\bdangerous\s*goods\b",
            r"\bundeclared\s*DG\b",
            r"\bmisdeclared\s*DG\b",
            r"\bDG\s*incident\b",
            r"\bdry\s*ice\b",
            r"\blithium\s*battery\b",
            r"\bflammable\b",
            r"\bcorrosive\b",
        ],
        "confidence": 0.90,
    },
]


# ============================================================
# 4. TEXT NORMALIZATION
# ============================================================


def normalize_text(text: object) -> str:
    """
    Normalize report text before rule matching.

    Important:
        This function keeps useful aviation acronyms but normalizes spacing,
        punctuation and common variants.
    """
    if text is None or pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Common normalization of spelling/format variants.
    replacements = {
        r"\bgo/a\b": "go around",
        r"\bgo-around\b": "go around",
        r"\btake-off\b": "takeoff",
        r"\btake off\b": "takeoff",
        r"\brejected take off\b": "rejected takeoff",
        r"\bwind shear\b": "windshear",
        r"\bbirdstrike\b": "bird strike",
        r"\bhigh-g\b": "high g",
        r"\btcas\s+ra\b": "TCAS RA",
        r"\bacas\s+ra\b": "ACAS RA",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Keep slash/hyphen minimally but normalize most punctuation to spaces.
    text = re.sub(r"[^A-Za-z0-9\s/\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 5. RULE ENGINE CLASS
# ============================================================


class RuleEngine:
    """
    MOR rule engine.

    Matching layers:
        1. Curated regex patterns for high-priority MOR triggers.
        2. Optional structured_MOR_rules.xlsx keyword matching.

    Output:
        RuleEngineResult containing rule_based_mor, matched_rule_ids,
        matched keywords, primary rule, and review hint.
    """

    def __init__(self, rules_file: Optional[Path] = None, use_structured_rules: bool = True):
        self.rules_file = rules_file or self._resolve_rules_file()
        self.use_structured_rules = use_structured_rules
        self.structured_rules_df = self._load_structured_rules() if use_structured_rules else pd.DataFrame()

    def _resolve_rules_file(self) -> Optional[Path]:
        """Resolve structured rules file path."""
        if DEFAULT_RULES_FILE.exists():
            return DEFAULT_RULES_FILE
        if FALLBACK_RULES_FILE.exists():
            return FALLBACK_RULES_FILE
        return None

    def _load_structured_rules(self) -> pd.DataFrame:
        """Load structured_MOR_rules.xlsx if available."""
        if self.rules_file is None or not self.rules_file.exists():
            return pd.DataFrame()

        try:
            df = pd.read_excel(self.rules_file, sheet_name="MOR_Rules", engine="openpyxl")
        except Exception:
            try:
                df = pd.read_excel(self.rules_file, engine="openpyxl")
            except Exception:
                return pd.DataFrame()

        # Ensure expected columns exist.
        for col in ["rule_id", "domain", "subcategory", "mor_condition", "keywords_or_triggers"]:
            if col not in df.columns:
                df[col] = ""

        df["rule_id"] = df["rule_id"].fillna("").astype(str)
        df["domain"] = df["domain"].fillna("").astype(str)
        df["subcategory"] = df["subcategory"].fillna("").astype(str)
        df["mor_condition"] = df["mor_condition"].fillna("").astype(str)
        df["keywords_or_triggers"] = df["keywords_or_triggers"].fillna("").astype(str)

        return df

    @staticmethod
    def _keyword_to_regex(keyword: str) -> str:
        """
        Convert plain keyword phrase to a lenient regex.

        Example:
            "bird strike" -> r"\bbird\s+strike\b"
        """
        keyword = keyword.strip()
        keyword = re.escape(keyword)
        keyword = keyword.replace(r"\ ", r"\s+")
        return rf"\b{keyword}\b"

    @staticmethod
    def _safe_regex_search(pattern: str, text: str) -> bool:
        """Safely evaluate regex pattern."""
        try:
            return re.search(pattern, text, flags=re.IGNORECASE) is not None
        except re.error:
            return False

    def _match_curated_patterns(self, normalized_text: str) -> List[RuleMatch]:
        """Match curated high-priority trigger patterns."""
        matches: List[RuleMatch] = []

        for rule in CURATED_MOR_PATTERNS:
            for pattern in rule["patterns"]:
                if self._safe_regex_search(pattern, normalized_text):
                    matches.append(
                        RuleMatch(
                            rule_id=rule["rule_id"],
                            source="curated_pattern",
                            domain=rule["domain"],
                            subcategory=rule["subcategory"],
                            mor_condition=rule["mor_condition"],
                            matched_keyword=pattern,
                            match_type="regex",
                            confidence=float(rule["confidence"]),
                        )
                    )
                    # Stop after first pattern match within the same curated rule.
                    break

        return matches

    def _match_structured_rules(self, normalized_text: str) -> List[RuleMatch]:
        """Match keyword triggers from structured_MOR_rules.xlsx."""
        matches: List[RuleMatch] = []

        if self.structured_rules_df.empty:
            return matches

        for _, row in self.structured_rules_df.iterrows():
            keywords_raw = str(row.get("keywords_or_triggers", "")).strip()
            if not keywords_raw:
                continue

            # structured_MOR_rules.xlsx uses comma-separated keyword triggers.
            keywords = [kw.strip() for kw in keywords_raw.split(",") if kw.strip()]

            for keyword in keywords:
                # Avoid very short/noisy keywords.
                if len(keyword) < 3:
                    continue

                pattern = self._keyword_to_regex(keyword.lower())
                if self._safe_regex_search(pattern, normalized_text.lower()):
                    matches.append(
                        RuleMatch(
                            rule_id=str(row.get("rule_id", "")),
                            source="structured_rules_excel",
                            domain=str(row.get("domain", "")),
                            subcategory=str(row.get("subcategory", "")),
                            mor_condition=str(row.get("mor_condition", "")),
                            matched_keyword=keyword,
                            match_type="keyword",
                            confidence=0.75,
                        )
                    )
                    # Stop after first keyword match for this structured rule.
                    break

        return matches

    @staticmethod
    def _deduplicate_matches(matches: List[RuleMatch]) -> List[RuleMatch]:
        """Remove duplicate rule IDs while keeping highest confidence match."""
        best_by_rule_id: Dict[str, RuleMatch] = {}

        for match in matches:
            key = match.rule_id or f"{match.source}_{match.matched_keyword}"
            if key not in best_by_rule_id or match.confidence > best_by_rule_id[key].confidence:
                best_by_rule_id[key] = match

        # Sort by confidence descending, curated first if tied.
        return sorted(
            best_by_rule_id.values(),
            key=lambda m: (m.confidence, m.source == "curated_pattern"),
            reverse=True,
        )

    def evaluate_report(self, report_text: object) -> Dict:
        """
        Evaluate a single report text.

        Returns a dictionary so it can be directly converted into a dataframe row.
        """
        normalized_text = normalize_text(report_text)

        matches: List[RuleMatch] = []
        matches.extend(self._match_curated_patterns(normalized_text))
        matches.extend(self._match_structured_rules(normalized_text))
        matches = self._deduplicate_matches(matches)

        if not matches:
            result = RuleEngineResult(
                rule_based_mor=0,
                rule_confidence=0.0,
                matched_rule_count=0,
                matched_rule_ids="",
                matched_keywords="",
                matched_domains="",
                matched_subcategories="",
                primary_rule_id=None,
                primary_domain=None,
                primary_subcategory=None,
                primary_condition=None,
                review_hint="No MOR rule trigger matched",
                matches=[],
            )
            return asdict(result)

        primary = matches[0]
        result = RuleEngineResult(
            rule_based_mor=1,
            rule_confidence=max(match.confidence for match in matches),
            matched_rule_count=len(matches),
            matched_rule_ids="; ".join(match.rule_id for match in matches),
            matched_keywords="; ".join(match.matched_keyword for match in matches),
            matched_domains="; ".join(sorted(set(match.domain for match in matches if match.domain))),
            matched_subcategories="; ".join(sorted(set(match.subcategory for match in matches if match.subcategory))),
            primary_rule_id=primary.rule_id,
            primary_domain=primary.domain,
            primary_subcategory=primary.subcategory,
            primary_condition=primary.mor_condition,
            review_hint="Rule engine indicates probable MOR",
            matches=[asdict(match) for match in matches],
        )
        return asdict(result)

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        prefix: str = "rule_",
    ) -> pd.DataFrame:
        """
        Apply rule engine to a dataframe.

        Args:
            df: Input dataframe.
            text_column: Column containing report text.
            prefix: Prefix to add to rule-engine output columns.

        Returns:
            DataFrame with rule-engine columns appended.
        """
        if text_column not in df.columns:
            raise ValueError(f"Text column not found in dataframe: {text_column}")

        output_df = df.copy()
        results = [self.evaluate_report(text) for text in output_df[text_column].tolist()]
        result_df = pd.DataFrame(results)

        # Do not expand nested matches JSON into many columns; keep it as JSON-like string.
        if "matches" in result_df.columns:
            result_df["matches"] = result_df["matches"].apply(lambda x: str(x))

        result_df = result_df.add_prefix(prefix)
        output_df = pd.concat([output_df.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)

        return output_df


# ============================================================
# 6. MODEL + RULE ENGINE COMBINATION LOGIC
# ============================================================


def combine_ml_and_rule_decision(row: pd.Series) -> pd.Series:
    """
    Combine ML prediction and rule-engine result for review decision.

    Required/expected columns:
        predicted_mor_label OR ml_mor_prediction
        prediction_confidence OR ml_confidence
        rule_rule_based_mor OR rule_based_mor

    Core business rule:
        If rule_engine says MOR and model says Non-MOR:
            Mandatory Review
    """
    # ML prediction label resolution.
    if "predicted_mor_label" in row.index:
        ml_pred = int(row["predicted_mor_label"])
    elif "ml_mor_prediction" in row.index:
        value = str(row["ml_mor_prediction"]).lower().strip()
        ml_pred = 1 if value in {"1", "mor", "yes", "true"} else 0
    else:
        ml_pred = 0

    # Rule result resolution.
    if "rule_rule_based_mor" in row.index:
        rule_value = row["rule_rule_based_mor"]
    elif "rule_based_mor" in row.index:
        rule_value = row["rule_based_mor"]
    else:
        rule_value = 0

    rule_mor = str(rule_value).lower().strip() in {"1", "true", "yes", "mor"}

    # Confidence resolution.
    if "prediction_confidence" in row.index and pd.notna(row["prediction_confidence"]):
        confidence = float(row["prediction_confidence"])
    elif "ml_confidence" in row.index and pd.notna(row["ml_confidence"]):
        confidence = float(row["ml_confidence"])
    else:
        confidence = 0.0

    final_recommendation = "MOR" if ml_pred == 1 else "Non-MOR"
    review_required = "No"
    review_priority = "None"
    review_reason = "No review rule triggered"

    # Mandatory rule requested by project.
    if rule_mor and ml_pred == 0:
        final_recommendation = "Human Review Required"
        review_required = "Yes"
        review_priority = "Mandatory"
        review_reason = "Rule engine indicates MOR but ML predicted Non-MOR"

    elif confidence < 0.70:
        review_required = "Yes"
        review_priority = "Required"
        review_reason = "ML prediction confidence below 0.70"

    elif ml_pred == 0 and confidence < 0.80:
        review_required = "Yes"
        review_priority = "Recommended"
        review_reason = "ML predicted Non-MOR with confidence below 0.80"

    return pd.Series({
        "final_recommendation": final_recommendation,
        "final_review_required": review_required,
        "final_review_priority": review_priority,
        "final_review_reason": review_reason,
    })


# ============================================================
# 7. STANDALONE TEST
# ============================================================


def main() -> None:
    """Quick standalone test for the rule engine."""
    engine = RuleEngine()

    sample_reports = [
        "Aircraft experienced bird strike during climb after departure.",
        "Smoke was observed in cabin during taxi.",
        "TCAS RA was triggered during cruise.",
        "Passenger requested wheelchair on arrival.",
        "Rejected takeoff was performed due to engine failure warning.",
        "Hydraulic leak observed after landing.",
    ]

    for report in sample_reports:
        result = engine.evaluate_report(report)
        print("\nReport:", report)
        print("Rule Based MOR:", result["rule_based_mor"])
        print("Confidence:", result["rule_confidence"])
        print("Matched Rule IDs:", result["matched_rule_ids"])
        print("Matched Keywords:", result["matched_keywords"])
        print("Primary Subcategory:", result["primary_subcategory"])
        print("Review Hint:", result["review_hint"])


if __name__ == "__main__":
    main()
