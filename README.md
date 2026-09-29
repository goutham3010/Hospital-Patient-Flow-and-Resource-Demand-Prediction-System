# 🏥 Hospital Patient Flow & Demand Prediction System
### *An AI-Powered Hospital Operations Decision Support System*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.44+-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/ML-Gradient%20Boosting%20%7C%20XGBoost-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![Explainable AI](https://img.shields.io/badge/XAI-TreeSHAP-orange.svg)](https://shap.readthedocs.io/)
[![Architecture](https://img.shields.io/badge/Architecture-RAG%20%2B%20Decision%20Support-purple.svg)]()

---

## 📌 Executive Summary
The **Hospital Patient Flow & Demand Prediction System** is a production-grade, 17-stage operational AI platform that transforms 120,000 longitudinal patient records into predictive intelligence. It forecasts next-day hospital admissions, anticipates department-level patient surges across General, ICU, HDU, and NICU wards, uncovers root-cause surge drivers via **Explainable AI (TreeSHAP)**, and retrieves actionable clinical protocols via **Retrieval-Augmented Generation (RAG)** to provide real-time decision support for nurse staffing, bed surge escalation, and elective surgery planning.

---

## 🏗️ End-to-End System Architecture (17-Stage Workflow)

```
                 ┌──────────────────────────────┐
                 │        Hospital Dataset       │
                 │        admissions.csv         │
                 │       120,000 records         │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │  01. Data Understanding      │
                 │                              │
                 │ • Dataset structure          │
                 │ • Features & data types      │
                 │ • Missing values             │
                 │ • Duplicate records           │
                 │ • Date range                 │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │       02. EDA                │
                 │                              │
                 │ • Admission trends           │
                 │ • Discharge trends            │
                 │ • Ward distribution          │
                 │ • Length of stay             │
                 │ • Hospital utilization       │
                 │ • Daily/weekly/monthly trends│
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ 03. Data Cleaning &          │
                 │     Preparation              │
                 │                              │
                 │ • Missing values             │
                 │ • Duplicates                 │
                 │ • Date conversion            │
                 │ • Invalid values             │
                 │ • Categorical encoding       │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ 04. Feature Engineering      │
                 │                              │
                 │ • Day of week                │
                 │ • Month                      │
                 │ • Weekend                    │
                 │ • Previous-day admissions    │
                 │ • 7-day admission average    │
                 │ • 14-day admission average   │
                 │ • Ward-level demand          │
                 │ • Length-of-stay features    │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ 05. Define Prediction Target │
                 │                              │
                 │   Future Patient Demand       │
                 │                              │
                 │ Example:                     │
                 │ Predict tomorrow's admissions│
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ 06. Train/Test Split         │
                 │                              │
                 │   Time-based split           │
                 │                              │
                 │ Historical → Training        │
                 │ Recent     → Testing         │
                 └──────────────┬───────────────┘
                                │
                                ▼
          ┌──────────────────────────────────────────┐
          │          07. ML Model Training           │
          │                                          │
          │  • Linear Regression                     │
          │  • Decision Tree Regressor               │
          │  • Random Forest Regressor               │
          │  • Gradient Boosting / XGBoost           │
          └────────────────────┬─────────────────────┘
                               │
                               ▼
                 ┌──────────────────────────────┐
                 │ 08. Model Evaluation         │
                 │                              │
                 │ • MAE                        │
                 │ • RMSE                       │
                 │ • R²                         │
                 │ • Actual vs Predicted        │
                 │ • Prediction trend           │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ 09. Hyperparameter Tuning    │
                 │                              │
                 │ • GridSearchCV               │
                 │ • RandomizedSearchCV         │
                 │ • Optimize final model       │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ 10. Explainable AI (SHAP)    │
                 │                              │
                 │ • Feature importance         │
                 │ • Global explanation         │
                 │ • Individual prediction      │
                 │ • Why demand increased       │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ 11. Save Final ML Model      │
                 │                              │
                 │ patient_flow_model.pkl       │
                 └──────────────┬───────────────┘
                                │
                                ▼
              ┌──────────────────────────────────────┐
              │       12. Streamlit Dashboard        │
              │                                      │
              │  • Hospital overview                  │
              │  • Current patient flow               │
              │  • Future demand prediction           │
              │  • Ward demand                        │
              │  • Charts & trends                    │
              │  • SHAP explanation                   │
              └──────────────────┬───────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │          13. LLM Assistant            │
              │                                      │
              │  ML prediction + SHAP results         │
              │                ↓                     │
              │             LLM                      │
              │                ↓                     │
              │  Human-readable explanation           │
              └──────────────────┬───────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │             14. RAG                  │
              │                                      │
              │ Hospital SOPs / Guidelines            │
              │                ↓                     │
              │       Document Retrieval             │
              │                ↓                     │
              │       Relevant Information            │
              │                ↓                     │
              │              LLM                     │
              └──────────────────┬───────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │       15. Decision Support            │
              │                                      │
              │  Predicted demand                    │
              │          +                           │
              │  Explanation                         │
              │          +                           │
              │  Hospital guidelines                 │
              │          ↓                           │
              │  Operational insights                │
              └──────────────────┬───────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │       16. Testing & Validation        │
              │                                      │
              │ • ML testing                          │
              │ • Data validation                     │
              │ • Dashboard testing                  │
              │ • LLM testing                        │
              │ • RAG testing                        │
              │ • End-to-end testing                 │
              └──────────────────┬───────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │        17. Deployment                 │
              │                                      │
              │ GitHub → Streamlit Cloud              │
              │                                      │
              │ Final Hospital Operations             │
              │ Decision Support System               │
              └──────────────────────────────────────┘
```

---

## 📂 Project Structure

```
Hospital Prediction System/
│
├── admissions.csv                      # Primary dataset (120,000 admission records)
│
├── src/                                # Core Engine Source Code
│   ├── __init__.py
│   ├── data_loader.py                  # Step 01: Dataset loading & structural inspection
│   ├── eda.py                          # Step 02: Exploratory data analysis & statistical summaries
│   ├── preprocessing.py                # Steps 03 & 04: Cleaning, daily series aggregation & feature engineering
│   ├── train.py                        # Steps 05-11: Time-based split, ML benchmarks, tuning, SHAP & persistence
│   ├── explainability.py               # Step 10: Local & Global TreeSHAP interpretability engine
│   ├── rag_engine.py                   # Step 14: Hospital SOP TF-IDF semantic retrieval engine
│   └── llm_assistant.py                # Steps 13 & 15: AI Clinical Operations Assistant & Decision Support
│
├── notebooks/                          # Interactive Jupyter Notebooks
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_Exploratory_Data_Analysis.ipynb
│   ├── 03_Data_Preprocessing_and_Features.ipynb
│   ├── 04_Model_Training_and_Tuning.ipynb
│   ├── 05_Explainable_AI_SHAP.ipynb
│   └── 06_RAG_and_Decision_Support.ipynb
│
├── models/                             # Saved Serialized Artifacts
│   ├── patient_flow_model.pkl          # Tuned model bundle with scalers, metadata & ward models
│   ├── metrics_summary.json            # MAE, RMSE, R², MAPE benchmark results
│   └── feature_importance.json         # TreeSHAP importance ranking
│
├── data/                               # Processed Data & Knowledge Base
│   ├── daily_patient_flow.csv          # Engineered daily time-series matrix (3,637 days)
│   ├── hospital_sops.json              # Curated hospital clinical & operational protocols
│   ├── test_predictions.csv           # Out-of-sample 2024 actual vs predicted records
│   └── eda_summary.json                # Precomputed statistical metrics
│
├── .streamlit/
│   └── config.toml                     # Custom glassmorphic dark theme configuration
│
├── streamlit_app.py                    # Step 12: Streamlit Interactive Command Dashboard
├── run_full_pipeline.py                # Master one-click end-to-end training & validation runner
├── test_system.py                      # Step 16: Automated unit & integration test suite
└── requirements.txt                    # Step 17: Production dependencies for deployment
```

---

## ⚡ Quick Start Guide

### 1. Prerequisites & Environment Setup
Clone the repository and install required packages:
```bash
python -m pip install -r requirements.txt
```

### 2. Execute Master Pipeline (One-Click)
Run data ingestion, EDA, feature engineering, model training, hyperparameter tuning, SHAP attribution, and validation:
```bash
python run_full_pipeline.py
```

### 3. Run Automated Tests
Verify all sub-systems (Data, Features, Inference, SHAP, RAG, Assistant):
```bash
python test_system.py
```

### 4. Launch the Interactive Decision Support Dashboard
```bash
streamlit run streamlit_app.py
```

---

## 📊 Model Benchmark Results (2024 Unseen Test Partition)

| Model Architecture | MAE (Patients) | RMSE | R² Score | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 4.451 | 5.573 | -0.0116 | 14.33% |
| **Decision Tree Regressor** | 4.485 | 5.669 | -0.0469 | 14.47% |
| **Random Forest Regressor** | 4.408 | 5.523 | 0.0065 | 14.19% |
| **Gradient Boosting (Tuned)** | **4.425** | **5.526** | **0.0053** | **14.22%** |

*Evaluation performed on 364 consecutive unseen days of operational hospital flow (2024).*

---

## 🧠 Explainable AI: Key Demand Drivers (SHAP)
1. **30-Day Rolling Inflow Trend (`admissions_roll_mean_30`)**: Captures seasonal baseline macro shifts.
2. **Short-Term Surge Momentum (`admissions_momentum_7_30`)**: Detects acute outbreaks or post-holiday backlogs.
3. **Discharge Velocity (`discharges_roll_mean_7`)**: Net hospital bed availability and turnover.
4. **Emergency Department Pressure (`type_emergency_lag_1`)**: Precursor indicator for acute hospital admissions.
5. **Active Bed Occupancy (`census_lag_1`)**: Capacity constraints gating elective intake.

---

## 📑 Clinical Decision Support & Hospital SOP Grounding (RAG)
When predicted demand or bed occupancy crosses critical thresholds, the system retrieves and triggers standardized operational workflows:
- **Level 0 (Green, < 85%)**: Routine staffing (ICU 1:1/1:2, HDU 1:2, General 1:5).
- **Level 1 (Yellow, 85% - 91.9%)**: Open 10 contingency flex beds, alert on-call nursing pool.
- **Level 2 (Amber, 92% - 97.9%)**: Mandate morning discharge rounds before 10:30 AM, transfer stable patients to Discharge Lounge, review elective admissions for postponement.
- **Level 3 (Red / Code Purple, >= 98%)**: Convene Hospital Incident Command System (HICS), divert non-critical ambulance transfers, cancel non-urgent elective surgeries.

---

## 🚀 Deployment (Streamlit Cloud & GitHub)
1. Commit and push repository to GitHub.
2. Link repository to [Streamlit Community Cloud](https://share.streamlit.io).
3. Set entrypoint file to `streamlit_app.py`.
4. Deploy with one click.
