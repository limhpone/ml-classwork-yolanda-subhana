from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# ── Load the full sklearn Pipeline once at startup ────────────────────────────
# model.pkl contains: SimpleImputer + StandardScaler + OneHotEncoder + LogisticRegression(C=5.0)
# Feature engineering (charges_per_month) is applied here before passing to the pipeline.
pipeline = pickle.load(open('model.pkl', 'rb'))

# Column order must match training exactly
NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_per_month']
CATEGORICAL_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]
ALL_COLS = NUMERICAL_COLS + CATEGORICAL_COLS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # ── Parse raw inputs ──────────────────────────────────────────────────────
    tenure          = float(data.get('tenure', 0))
    monthly_charges = float(data.get('MonthlyCharges', 0))
    total_charges   = float(data.get('TotalCharges', 0))

    # ── Feature engineering (must mirror train.ipynb exactly) ─────────────────
    # charges_per_month: normalises total spend by tenure length
    # +1 prevents division-by-zero for new customers with tenure=0
    charges_per_month = total_charges / (tenure + 1)

    # ── Build single-row DataFrame ────────────────────────────────────────────
    row = {
        'tenure':           tenure,
        'MonthlyCharges':   monthly_charges,
        'TotalCharges':     total_charges,
        'charges_per_month': charges_per_month,
    }
    for col in CATEGORICAL_COLS:
        row[col] = str(data.get(col, ''))

    df = pd.DataFrame([row], columns=ALL_COLS)

    # ── Inference ─────────────────────────────────────────────────────────────
    # Pipeline handles: imputation → scaling (numerical) + encoding (categorical) → LR
    prediction  = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0]

    # ── Map 0/1 back to human-readable label ──────────────────────────────────
    label = 'Churn' if prediction == 1 else 'No Churn'

    return jsonify({
        'prediction':    int(prediction),
        'label':         label,
        'confidence':    round(float(max(probability)) * 100, 2),
        'prob_churn':    round(float(probability[1]) * 100, 2),
        'prob_no_churn': round(float(probability[0]) * 100, 2),
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
