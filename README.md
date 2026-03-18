# Telco Customer Churn Prediction

**Course:** Machine Learning

**Authors:** Aye Khin Khin Hpone (Yolanda Lim) — 125970 | Subhana Chitrakar — 126138

---

## Deployment Evidence

### Live Deployment

- **Yolanda Lim (ST125970):** <http://192.41.170.112:5970>  
- **Subhana Chitrakar (ST126138):** <http://192.41.170.112:6138>  

### Deployed Model

**Telco Churn Predictor**  
Aye Khin Khin Hpone (Yolanda Lim) · ST125970 · Subhana Chitrakar · ST126138  
Logistic Regression Pipeline · Machine Learning

### Video Recording (Deployed ML App)

#### Watch Online
- **YouTube demo:** <https://youtu.be/QBTj7j6pSYs>

#### Local Video File
- **Download / Stream:** [125970_YolandaLim_Aye Khin Khin Hpone.mp4](screencapture_ml_yolanda_deployed/125970_YolandaLim_Aye%20Khin%20Khin%20Hpone.mp4)

#### Video Player

<video width="100%" controls>
  <source src="screencapture_ml_yolanda_deployed/125970_YolandaLim_Aye Khin Khin Hpone.mp4" type="video/mp4">
  Your browser does not support the video tag. Download the file above or watch on YouTube.
</video>

### Deployment Screenshots (9)

<table>
   <tr>
      <td><img src="screencapture_ml_yolanda_deployed/deploy.png" alt="Deployment screenshot 1" width="100%" /></td>
      <td><img src="screencapture_ml_yolanda_deployed/homepage.png" alt="Deployment screenshot 2" width="100%" /></td>
      <td><img src="screencapture_ml_yolanda_deployed/homepage1_with%20deploymentadd.png" alt="Deployment screenshot 3" width="100%" /></td>
   </tr>
   <tr>
      <td><img src="screencapture_ml_yolanda_deployed/Screenshot%202026-03-18%20124645.png" alt="Deployment screenshot 4" width="100%" /></td>
      <td><img src="screencapture_ml_yolanda_deployed/Screenshot%202026-03-18%20124657.png" alt="Deployment screenshot 5" width="100%" /></td>
      <td><img src="screencapture_ml_yolanda_deployed/Screenshot%202026-03-18%20124708.png" alt="Deployment screenshot 6" width="100%" /></td>
   </tr>
   <tr>
      <td><img src="screencapture_ml_yolanda_deployed/Screenshot%202026-03-18%20124719.png" alt="Deployment screenshot 7" width="100%" /></td>
      <td><img src="screencapture_ml_yolanda_deployed/Screenshot%202026-03-18%20124729.png" alt="Deployment screenshot 8" width="100%" /></td>
      <td><img src="screencapture_ml_yolanda_deployed/Screenshot%202026-03-18%20124740.png" alt="Deployment screenshot 9" width="100%" /></td>
   </tr>
</table>

---

## Model Performance at a Glance

| Metric | Value |
| -------- | ----- |
| Accuracy | 80.77% |
| AUC-ROC | 0.8468 |
| CV AUC (5-fold) | 0.8482 ± 0.012 |
| Best Model | Logistic Regression (C=5.0) |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Dataset](#3-dataset)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis-eda)
5. [Feature Engineering](#5-feature-engineering)
6. [Preprocessing Pipeline](#6-preprocessing-pipeline)
7. [Model Training & Results](#7-model-training--results)
8. [Flask Web Application](#8-flask-web-application)
9. [How to Run](#9-how-to-run)
10. [Analysis Summary & Business Insights](#10-analysis-summary--business-insights)

---

## 1. Project Overview

This project builds an end-to-end machine learning classification system to predict customer churn for a telecommunications company. Using the IBM Telco Customer Churn dataset (7,043 customers, 20 features), it trains a Logistic Regression pipeline and deploys it as an interactive Flask web application containerised with Docker.

Customer churn is a critical business metric — acquiring new customers costs 5–10× more than retaining existing ones. An accurate predictor allows the company to:

- Identify at-risk customers before they cancel
- Target personalised retention offers to high-risk segments
- Understand the key drivers of churn through model coefficients
- Reduce revenue loss from preventable subscriber departures

---

## 2. Project Structure

```text
ml-project/
├── train_EDA.ipynb                    # EDA, feature engineering, model training, evaluation
├── test_predict.ipynb                 # 6 inference tests covering edge cases and batch prediction
├── app.py                             # Flask REST API backend
├── templates/
│   └── index.html                     # Interactive web UI with probability bars
├── model.pkl                          # Serialised sklearn Pipeline (preprocessor + classifier)
├── requirements.txt                   # Pinned Python dependencies
├── Dockerfile                         # Container definition for deployment
├── .gitignore
└── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset (not committed in production)
```

## 3. Dataset

**Source:** IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)

- **Rows:** 7,043 customer records
- **Columns:** 21 (20 features + 1 target)
- **Target:** `Churn` (Yes → 1, No → 0)
- **Class distribution:** 73.5% No Churn (5,174) | 26.5% Churn (1,869) — moderately imbalanced

---

- Kaggle Dataset Link: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

### Features

| Category | Features |
| --------- | -------- |
| Demographics (4) | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account / Billing (6) | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Phone Services (2) | `PhoneService`, `MultipleLines` |
| Internet Services (7) | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Engineered (1) | `charges_per_month = TotalCharges / (tenure + 1)` — derived, not in raw CSV |
| Target (1) | `Churn` (Yes / No → 1 / 0) |

### Data Quality Fixes

| Issue | Rows Affected | Fix |
| ----- | ------------- | --- |
| `TotalCharges` stored as string (spaces) | 11 rows (tenure=0) | `pd.to_numeric(errors='coerce')` then `fillna(median)` |
| `SeniorCitizen` encoded as integer 0/1 | All rows | Cast to string — treated as categorical by `OneHotEncoder` |
| `customerID` — unique identifier, no signal | All rows | Dropped before training |

---

## 4. Exploratory Data Analysis (EDA)

### Key EDA Findings

- **Contract type** is the strongest categorical predictor — month-to-month customers churn at ~43% vs ~11% (one year) and ~3% (two year)
- **Fiber optic** internet users churn significantly more than DSL or no-internet customers
- **Electronic check** payment method is associated with higher churn; autopay customers stay longer
- Customers **without OnlineSecurity or TechSupport** churn more
- `TotalCharges` is highly collinear with `tenure` — the `charges_per_month` feature disentangles this

---

## 5. Feature Engineering

One new numerical feature is derived and validated to improve AUC by **+0.47 percentage points** over the baseline:

| Feature | Formula | Rationale |
| ------- | ------- | --------- |
| `charges_per_month` | `TotalCharges / (tenure + 1)` | Normalises total spend by tenure. The `+1` prevents division-by-zero for `tenure=0` (new customers). Churners tend to have high charges relative to their tenure — this makes that signal explicit to the model. |

> **Note:** `charges_per_month` is computed in both `train_EDA.ipynb` and `app.py` using the same formula. These must stay in sync — changing one without the other will silently degrade predictions.

---

## 6. Preprocessing Pipeline

A single `sklearn.Pipeline` encapsulates all transformations and the classifier, guaranteeing no data leakage and clean serialisation into `model.pkl`:

| Step | Transformer | Applied To | Why |
| ---- | ----------- | --------- | --- |
| 1 — Impute (num) | `SimpleImputer(strategy='median')` | All numerical columns | Handles 11 NaN rows from `TotalCharges` coercion |
| 2 — Scale | `StandardScaler` | All numerical columns | Logistic regression is sensitive to feature magnitude |
| 3 — Impute (cat) | `SimpleImputer(strategy='most_frequent')` | All 16 categorical columns | Safety net for missing values at inference |
| 4 — Encode | `OneHotEncoder(drop='first', handle_unknown='ignore')` | All 16 categorical columns | `drop='first'` prevents multicollinearity |
| 5 — Classify | `LogisticRegression(C=5.0, max_iter=2000)` | Transformed feature matrix | Best AUC-ROC; C tuned by GridSearchCV |

The full pipeline is saved as `model.pkl`. `app.py` only needs to call `pipeline.predict(df)` — no manual scaling or encoding is needed at inference time.

---

## 7. Model Training & Results

### Models Compared

| Model | Accuracy | AUC-ROC | F1 (Churn) | CV AUC (5-fold) |
| ----- | -------- | ------- | --------- | -------------- |
| **Logistic Regression ✓** | **0.8077** | **0.8468** | **0.5973** | **0.8491 ± 0.012** |
| Decision Tree | 0.7942 | 0.8295 | 0.5845 | 0.8289 ± 0.089 |
| Random Forest | 0.7807 | 0.8236 | 0.5422 | 0.8266 ± 0.0116 |
| KNN | 0.7658 | 0.7934 | 0.5528 | 0.7829 ± 0.092 |

### Why Logistic Regression

- Highest AUC-ROC (0.8468) — primary metric for imbalanced binary classification
- Lowest variance in 5-fold CV (± 0.012) — most generalisable
- Hyperparameter `C` tuned via `GridSearchCV` across `[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]`; C=5.0 selected
- Coefficients are interpretable — top churn drivers visible in the feature importance chart
- Fast inference — suitable for real-time Flask API

### Final Classification Report (20% Hold-out Test Set)

```text
              precision    recall  f1-score   support

    No Churn       0.84      0.91      0.87      1035
       Churn       0.67      0.54      0.60       374

    accuracy                           0.81      1409
   macro avg       0.76      0.72      0.74      1409
weighted avg       0.80      0.81      0.80      1409
```

### Threshold Optimisation

| Threshold | Accuracy | F1 (Churn) | Recall (Churn) | Notes |
| --------- | -------- | --------- | ------------- | ----- |
| 0.500 (deployed) | 0.8077 | 0.5973 | 0.54 | Balanced precision/recall; used in deployment |
| 0.323 (optimal F1) | 0.7715 | 0.6333 | 0.74 | Higher recall; lower overall accuracy |

> **Deployment decision:** threshold=0.5 is used in `model.pkl`. To maximise churn recall (catching more at-risk customers), apply threshold=0.323 in `app.py` by replacing `pipeline.predict(df)` with `(pipeline.predict_proba(df)[:,1] >= 0.323).astype(int)`.

---

## 8. Flask Web Application

### API Endpoints

| Route | Method | Description |
| ----- | ------ | ----------- |
| `GET /` | GET | Serve the prediction UI (`index.html`) |
| `POST /predict` | POST | Predict churn for one customer — accepts JSON, returns prediction + probabilities |

### Request (POST /predict)

Send a JSON body with all 19 raw feature fields (same names as the CSV columns, excluding `customerID` and `Churn`). `charges_per_month` is computed automatically in `app.py`.

### Response

```json
{
  "prediction":    1,
  "label":         "Churn",
  "confidence":    65.5,
  "prob_churn":    65.5,
  "prob_no_churn": 34.5
}
```

- `prediction`: `0` (No Churn) or `1` (Churn)
- `label`: human-readable string mapped from `prediction`
- `confidence`: probability of the predicted class × 100
- `prob_churn` / `prob_no_churn`: both class probabilities

### Feature Engineering in app.py

```python
tenure            = float(data['tenure'])
total_charges     = float(data['TotalCharges'])
charges_per_month = total_charges / (tenure + 1)   # must match train_EDA.ipynb
```

---

## 9. How to Run

### Source Code

GitHub: <https://github.com/limhpone/ml-classwork-yolanda-subhana>  
Docker Hub: <https://hub.docker.com/repository/docker/yolandalim/ml-yolanda-subhana/general>  

```bash
git clone https://github.com/limhpone/ml-classwork-yolanda-subhana.git
cd ml-project

python -m venv .venv
# Activate (Windows)
.venv\Scripts\activate
# Activate (macOS/Linux)
source .venv/bin/activate

pip install -r requirements.txt
python app.py

docker build -t ml-yolanda-subhana .
docker run -p 5000:5000 ml-yolanda-subhana
```

### Option C — Web UI Walkthrough

Once the server is running :

1. Fill in the customer profile using the dropdown menus and numeric inputs across all four sections (Demographics, Account Information, Phone Services, Internet Services)
2. Click **Predict Churn**
3. The result panel shows:
   - A green ✅ "likely to STAY" or red ⚠️ "likely to CHURN" headline
   - Animated probability bars for both Churn % and No Churn %

---
## 10. Analysis Summary & Business Insights

### Key Findings

- **Contract type is the #1 churn driver.** Month-to-month customers churn at ~43% vs ~3% for two-year contracts. Upgrading customers to longer contracts is the single most impactful retention lever.
- **New customers (0–12 months) are the highest-risk group.** The 0–12 month tenure band has >47% churn rate. Early engagement programmes (onboarding calls, loyalty rewards) could significantly reduce attrition.
- **Fiber optic internet is associated with higher churn.** This may indicate value-for-money concerns. Pricing review or service quality improvements could help.
- **Security and support add-ons are protective.** Customers with `OnlineSecurity` and `TechSupport` churn less. Offering these as free trial bundles to at-risk customers could improve retention.
- **Electronic check payers churn more.** Incentivising autopay enrolment (e.g. a discount) could reduce churn in this segment.

---

Aye Khin Khin Hpone (Yolanda Lim) — ST125970 | Subhana Chitrakar — ST126138 | Machine Learning Course
