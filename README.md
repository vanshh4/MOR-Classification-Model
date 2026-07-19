# MOR Classification Model

> **Internship Project by Vansh Bhardwaj under IndiGo - Flight Safety Department**

## 1. Project Overview

The **MOR Classification Model** is a machine learning and rule-assisted classification system developed as an internship project under **IndiGo's Flight Safety Department**.

The objective of this project is to assist the Flight Safety team in identifying whether a newly reported aviation safety occurrence should be classified as a **Mandatory Occurrence Report (MOR)** or **Non-MOR / VSR**, based on historical occurrence data, structured DGCA reporting records, and predefined MOR rule conditions.

The system accepts a safety report written in normal English and generates:

- MOR / Non-MOR prediction
- Prediction confidence score
- Rule-based MOR trigger match
- MOR category / sub-category prediction
- Final review requirement
- Review priority and review reason

This project is intended to work as a **decision-support and prioritization tool** for Flight Safety workflows. It is **not intended to replace final human judgement, DGCA reporting responsibility, or safety officer review**.

---

## 2. Project Intention and Use Case

Flight Safety departments receive reports from pilots, crew, ground staff, and other operational personnel. These reports may describe events such as bird strikes, smoke or fumes, rejected takeoffs, runway incursions, hard landings, TCAS RA events, ground collisions, technical defects, and other safety-related occurrences.

The key operational challenge is to efficiently identify which reports may qualify as MORs and require mandatory reporting or deeper safety review.

This project addresses that challenge by combining:

1. **Machine Learning Classification**
   - Learns patterns from historical DGCA occurrence records.
   - Predicts whether a report is likely to be MOR or Non-MOR.

2. **Rule-Based MOR Validation**
   - Checks report text against structured MOR rules and high-priority safety trigger conditions.
   - Ensures that critical events such as bird strike, smoke, fire, TCAS RA, rejected takeoff, and runway incursion are not missed only because of ML uncertainty.

3. **Category / Sub-category Prediction**
   - Predicts likely MOR category, event type, and report title/sub-category for reports identified as MOR or rule-triggered MOR.

4. **Human Review Workflow**
   - Flags uncertain or conflicting predictions for human review.
   - Especially prioritizes cases where the rule engine says MOR but the ML model predicts Non-MOR.

---

## 3. Current Model Capabilities

The system currently supports the following capabilities:

- Preprocessing of Volrep pilot-report Excel files.
- Preprocessing of Occurrence Master Sheet for supervised ML training.
- Structured MOR rules generation from MOR rulebook conditions.
- MOR binary model training using TF-IDF + Logistic Regression.
- MOR category and sub-category model training.
- Rule-based MOR trigger matching.
- Unified ML + rules + category prediction pipeline.
- Streamlit UI for single-report prediction.
- Review queue generation for safety officer inspection.
- Threshold tuning and threshold comparison for MOR detection.

---

## 4. Tech Stack

### Programming Language

- Python 3.11 / 3.12

### Data Processing

- pandas
- numpy
- openpyxl

### Machine Learning

- scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- joblib

### Visualization / Evaluation

- matplotlib
- Excel-based evaluation reports

### Web Interface

- Streamlit
- Custom CSS inside Streamlit for professional UI styling

### Frontend Demonstration Page

- HTML
- CSS
- JavaScript

### Storage Formats

- Excel `.xlsx`
- Pickle / Joblib `.pkl`
- Markdown `.md`

---

## 5. Model Specification

### 5.1 MOR Binary Classifier

The binary classifier predicts whether a report is:

```text
MOR = 1
Non-MOR / VSR = 0
```

#### Input Feature

```text
clean_brief_description
```

#### Target Label

```text
mor_label
```

#### Dataset Used

```text
data/processed/training/mor_binary_training_dataset.xlsx
```

#### Algorithm

```text
TF-IDF Vectorizer + Logistic Regression
```

#### Split Strategy

```text
group_based_split
```

The group-based split is used to reduce data leakage caused by duplicate or near-duplicate occurrence summaries.

#### Main Output Files

```text
models/mor_binary_model.pkl
models/mor_binary_vectorizer.pkl
outputs/evaluation/mor_binary_evaluation.xlsx
outputs/evaluation/mor_binary_confusion_matrix.png
```

#### Key Evaluation Metrics

The binary model tracks:

- Accuracy
- Precision for MOR
- Recall for MOR
- F1-score for MOR
- Macro F1-score
- False Negatives
- False Positives
- Confusion Matrix

For this project, **Recall for MOR** and **False Negatives** are the most important metrics because missing an actual MOR is a higher-risk error than sending an extra case for human review.

---

### 5.2 MOR Category / Sub-category Classifiers

The category models classify MOR reports into category-level labels.

#### Dataset Used

```text
data/processed/training/mor_category_training_dataset.xlsx
```

#### Input Feature

```text
clean_brief_description
```

#### Models Trained

```text
1. mor_category_level1
   Target: iata_level_1_normalized

2. mor_event_type
   Target: iata_event_level_3_normalized

3. mor_report_title
   Target: report_title_normalized
```

#### Algorithm

```text
TF-IDF Vectorizer + Logistic Regression
```

#### Main Output Files

```text
models/mor_category_level1_model.pkl
models/mor_category_level1_vectorizer.pkl
models/mor_event_type_model.pkl
models/mor_event_type_vectorizer.pkl
models/mor_report_title_model.pkl
models/mor_report_title_vectorizer.pkl
outputs/evaluation/mor_category_evaluation.xlsx
```

---

### 5.3 Rule Engine

The rule engine checks report text against predefined MOR conditions.

#### Main Script

```text
src/rule_engine.py
```

#### Rule Source

```text
data/processed/rules/structured_MOR_rules.xlsx
```

#### Key MOR Conditions Checked

The rule engine checks for high-priority MOR trigger patterns such as:

- Bird strike
- Smoke
- Fire
- Engine shutdown
- Rejected takeoff
- Runway incursion
- Hard landing
- Balked landing
- TCAS RA
- Windshear
- Fuel leak
- Hydraulic failure
- Ground collision
- Dangerous goods

#### Key Rule Logic

```text
If rule_engine says MOR and ML model says Non-MOR:
    Mandatory Review
```

This helps reduce the risk of missing critical safety events.

---

### 5.4 Unified Prediction Pipeline

The unified prediction pipeline combines:

```text
MOR Binary Model
+ Rule Engine
+ Category Models
+ Review Logic
```

#### Main Script

```text
src/predict_mor_with_rules.py
```

#### Input

Single report text or processed Volrep Excel file.

#### Output

```text
MOR / Non-MOR prediction
MOR probability
Prediction confidence
Rule-based MOR match
Matched rule IDs
Matched keywords
Predicted category
Predicted event type
Predicted report title
Final recommendation
Review required
Review priority
Review reason
```

---

## 6. Review Logic

The current review logic is:

```text
If prediction confidence < 0.70:
    Human Review Required

If predicted Non-MOR but confidence < 0.80:
    Human Review Recommended

If rule engine says MOR but ML model says Non-MOR:
    Mandatory Review
```

This logic ensures that uncertain or rule-conflicting cases are not silently accepted.

---

## 7. Project Folder Structure

```text
MOR_Classification_Project/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   │
│   ├── raw/
│   │   │
│   │   ├── occurrence/
│   │   │   ├── Occurrence Sheet 2026-MASTER.xlsx
│   │   │   └── archive/
│   │   │       └── Occurrence Sheet 2026-MASTER_previous.xlsx
│   │   │
│   │   ├── volrep/
│   │   │   ├── Volrep_Processed (Jan-Mar).xlsx
│   │   │   └── Volrep_Processed (APR-JUN).xlsx
│   │   │
│   │   └── rules/
│   │       └── Rules for Reportable Occurences.pdf
│   │
│   └── processed/
│       │
│       ├── occurrence/
│       │   └── occurrence_master_cleaned.xlsx
│       │
│       ├── training/
│       │   ├── mor_binary_training_dataset.xlsx
│       │   └── mor_category_training_dataset.xlsx
│       │
│       ├── volrep/
│       │   ├── combined_volrep_2026H1_cleaned.xlsx
│       │   └── combined_volrep_2026H1_refined.xlsx
│       │
│       └── rules/
│           └── structured_MOR_rules.xlsx
│
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── script.js
│
├── models/
│   ├── mor_binary_model.pkl
│   ├── mor_binary_vectorizer.pkl
│   ├── mor_category_level1_model.pkl
│   ├── mor_category_level1_vectorizer.pkl
│   ├── mor_event_type_model.pkl
│   ├── mor_event_type_vectorizer.pkl
│   ├── mor_report_title_model.pkl
│   └── mor_report_title_vectorizer.pkl
│
├── outputs/
│   │
│   ├── evaluation/
│   │   ├── mor_binary_evaluation.xlsx
│   │   ├── mor_binary_confusion_matrix.png
│   │   └── mor_category_evaluation.xlsx
│   │
│   ├── predictions/
│   │   └── volrep_mor_predictions_with_rules.xlsx
│   │
│   └── review_queue/
│       └── volrep_review_queue.xlsx
│
├── src/
│   ├── preprocess_occurrence.py
│   ├── preprocess_volrep.py
│   ├── refine_combined_volrep.py
│   ├── structured_rules_generation.py
│   ├── train_mor_binary.py
│   ├── train_mor_category.py
│   ├── rule_engine.py
│   └── predict_mor_with_rules.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 8. Important Data Confidentiality Note

The `data/` folder contains confidential operational safety data, occurrence reports, Volrep reports, DGCA reporting information, and rule documents.

Therefore, the complete `data/` folder must be added to `.gitignore` and must **never** be pushed to GitHub or any public repository.

Recommended `.gitignore` entry:

```gitignore
# Confidential aviation safety data
data/

# Trained model artifacts may also be restricted depending on policy
models/

# Generated outputs and review queues
outputs/

# Python cache
__pycache__/
*.pyc

# Environment files
.env
.venv/
venv/

# OS files
.DS_Store
Thumbs.db
```

If model files are allowed to be versioned internally, remove `models/` from `.gitignore`. However, for public repositories, both `data/` and `models/` should be treated carefully.

---

## 9. Installation

Create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install pandas numpy openpyxl scikit-learn joblib matplotlib streamlit
```

Alternatively, if `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

---

## 10. Recommended Run Order

### Step 1: Preprocess Occurrence Master

```bash
python src/preprocess_occurrence.py
```

Generates:

```text
data/processed/occurrence/occurrence_master_cleaned.xlsx
data/processed/training/mor_binary_training_dataset.xlsx
data/processed/training/mor_category_training_dataset.xlsx
```

---

### Step 2: Preprocess Volrep Reports

```bash
python src/preprocess_volrep.py
```

Generates:

```text
data/processed/volrep/combined_volrep_2026H1_cleaned.xlsx
```

---

### Step 3: Refine Combined Volrep File

```bash
python src/refine_combined_volrep.py
```

Generates:

```text
data/processed/volrep/combined_volrep_2026H1_refined.xlsx
```

---

### Step 4: Generate Structured MOR Rules

```bash
python src/structured_rules_generation.py
```

Generates:

```text
data/processed/rules/structured_MOR_rules.xlsx
```

---

### Step 5: Train MOR Binary Model

```bash
python src/train_mor_binary.py
```

Generates:

```text
models/mor_binary_model.pkl
models/mor_binary_vectorizer.pkl
outputs/evaluation/mor_binary_evaluation.xlsx
outputs/evaluation/mor_binary_confusion_matrix.png
```

---

### Step 6: Train MOR Category Models

```bash
python src/train_mor_category.py
```

Generates:

```text
models/mor_category_level1_model.pkl
models/mor_category_level1_vectorizer.pkl
models/mor_event_type_model.pkl
models/mor_event_type_vectorizer.pkl
models/mor_report_title_model.pkl
models/mor_report_title_vectorizer.pkl
outputs/evaluation/mor_category_evaluation.xlsx
```

---

### Step 7: Run Unified ML + Rule Prediction on Volrep Reports

```bash
python src/predict_mor_with_rules.py --input data/processed/volrep/combined_volrep_2026H1_refined.xlsx --text-column clean_report_text --output outputs/predictions/volrep_mor_predictions_with_rules.xlsx
```

Generates:

```text
outputs/predictions/volrep_mor_predictions_with_rules.xlsx
```

---

### Step 8: Run Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

The UI supports single-report prediction with:

- MOR / Non-MOR output
- Confidence score
- Rule match
- Category / sub-category
- Review requirement

---

## 11. Streamlit Application

The Streamlit application is located at:

```text
app/streamlit_app.py
```

It provides a professional Flight Safety interface where a user can paste a report and receive:

```text
ML Prediction
Prediction Confidence
Rule Engine Match
Predicted Category
Predicted Event Type
Predicted Report Title
Final Review Requirement
Technical Prediction Details
```

This interface is intended for demonstration and internal review support.

---

## 12. Frontend Explanation Page

The project also contains a frontend-only webpage under:

```text
frontend/
```

This webpage is not the active ML prediction interface. It is used to visually explain:

- Project intent
- Developer workflow
- Data flow
- Preprocessing pipeline
- Model training flow
- Rule engine integration
- Human review feedback loop

---

## 13. Output Interpretation

### MOR Prediction

```text
MOR
```

The report is likely a Mandatory Occurrence Report.

```text
Non-MOR
```

The report is likely not an MOR based on the ML model.

### Rule-Based MOR

```text
rule_based_mor = 1
```

The rule engine found a safety-rule trigger.

```text
rule_based_mor = 0
```

No configured MOR rule trigger was found.

### Review Priority

```text
Mandatory
```

High-priority review required, especially when the rule engine says MOR but ML says Non-MOR.

```text
Required
```

Prediction confidence is below the acceptable threshold.

```text
Recommended
```

Model predicted Non-MOR but confidence is not strong enough.

```text
None
```

No review rule was triggered.

---

## 14. Key Limitations

- The first model is trained primarily on structured Occurrence Master brief descriptions, while Volrep reports are longer pilot narratives.
- Category prediction depends on the quality and distribution of category labels in the training dataset.
- Rare MOR categories may have limited examples and may require manual validation.
- Rule matching is keyword/phrase driven and may require continued refinement.
- The system should not be used as an autonomous regulatory reporting decision-maker.

---

## 15. Future Improvements

Recommended future enhancements include:

- Match Volrep reports with Occurrence Master records to create a verified pilot-narrative training dataset.
- Retrain the binary classifier using verified Volrep labels.
- Improve category prediction using more balanced examples.
- Add threshold optimization directly into the training pipeline.
- Add model versioning under `models/v1`, `models/v2`, etc.
- Add batch upload support in the Streamlit UI.
- Add reviewer feedback capture directly in the UI.
- Add authentication if the application is used beyond local development.
- Add Power BI dashboard for prediction trends and review queue monitoring.

---

## 16. Project Status

Current completed components:

- Occurrence preprocessing
- Volrep preprocessing
- Refined Volrep dataset
- Structured MOR rule extraction
- MOR binary classifier
- MOR category classifiers
- Rule engine
- ML + rule unified prediction pipeline
- Streamlit single-report prediction UI
- Threshold tuning and comparison

Current pending / recommended next components:

- Manual review of Volrep prediction output
- Volrep-to-Occurrence matching
- Human-verified Volrep training dataset
- Model retraining using verified Volrep labels
- Final project documentation and presentation material

---

## 17. Disclaimer

This project is an internship project developed by **Vansh Bhardwaj under IndiGo's Flight Safety Department**.

The model is intended to assist with safety report triage, prioritization, and review support. Final MOR classification and regulatory reporting decisions must remain with authorized Flight Safety personnel and applicable organizational procedures.
