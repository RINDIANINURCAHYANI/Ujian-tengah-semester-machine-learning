from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import plotly
import plotly.graph_objs as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, accuracy_score
)

import io
import base64

app = Flask(__name__)

# =============================
# 1. LOAD DATASET ASLI KAMU
# =============================
DATA_PATH = "diabetes_prediction.csv"   # pastikan file ini ada di folder yang sama
df = pd.read_csv(DATA_PATH)

# Kolom sesuai dataset asli
FEATURES = [
    'gender', 'age', 'hypertension', 'heart_disease',
    'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'
]

TARGET = 'diabetes'

# =============================
# 2. PERSIAPAN DATA & TRAIN LOGISTIC REGRESSION
# =============================
X_raw = df[FEATURES]
y = df[TARGET]

# Encode kategorikal (gender & smoking_history)
X = pd.get_dummies(X_raw, drop_first=True)

# Train-test split (pakai stratify supaya distribusi label seimbang)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Model (class_weight='balanced' membantu karena kelas 1 lebih sedikit)
model = LogisticRegression(max_iter=500, class_weight='balanced')
model.fit(X_train, y_train)

# Simpan & load model (opsional, supaya konsisten)
joblib.dump(model, "model_logistic.pkl")
model = joblib.load("model_logistic.pkl")


# =============================
# FUNGSI BANTU: FIGURE -> BASE64
# =============================

def fig_to_base64():
    """Simpan figure aktif matplotlib ke PNG base64 string (tanpa file)."""
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()
    return img_base64


def generate_dataset_plots():
    """Buat semua grafik EDA (countplot, histplot, heatmap) dalam bentuk base64."""
    df_viz = df.copy()
    df_viz["gender"] = df_viz["gender"].str.replace("Other", "Female")

    # 1) Distribusi Target (label diabetes)
    plt.figure(figsize=(6, 4))
    sns.countplot(x='diabetes', data=df_viz, palette='Set2')
    plt.title("Distribusi Label Diabetes (0 = Tidak, 1 = Ya)")
    plt.xlabel("Diabetes")
    plt.ylabel("Jumlah")
    diabetes_plot = fig_to_base64()

    # 2) Distribusi Gender
    plt.figure(figsize=(6, 4))
    sns.countplot(x='gender', data=df_viz, palette='Blues')
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    gender_plot = fig_to_base64()

    # 3) Distribusi Smoking History
    plt.figure(figsize=(6, 4))
    sns.countplot(x='smoking_history', data=df_viz, palette='Blues')
    plt.xlabel('Smoking History')
    plt.ylabel('Count')
    plt.title('Smoking History Distribution')
    plt.xticks(rotation=30)
    smoking_plot = fig_to_base64()

    # 4) Distribusi Umur (histplot)
    plt.figure(figsize=(6, 4))
    sns.histplot(df_viz['age'], kde=True, color='steelblue')
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    age_plot = fig_to_base64()

    # 5) Heatmap Korelasi
    num_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level",
                "hypertension", "heart_disease", "diabetes"]

    corr_matrix = df_viz[num_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(num_cols)))
    ax.set_yticks(np.arange(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="left")
    ax.set_yticklabels(num_cols)

    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            c = corr_matrix.iloc[i, j]
            ax.text(
                j, i, f"{c:.2f}",
                va="center", ha="center",
                color="black" if abs(c) < 0.5 else "white"
            )

    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    corr_heatmap_plot = fig_to_base64()

    return {
        "diabetes_plot": diabetes_plot,
        "gender_plot": gender_plot,
        "smoking_plot": smoking_plot,
        "age_plot": age_plot,
        "corr_heatmap_plot": corr_heatmap_plot,
    }


# Buat sekali saat server start
dataset_plots = generate_dataset_plots()


# =============================
# 3. HALAMAN UTAMA (1-based index)
# =============================
@app.route("/")
def home():
    dataset_options = list(range(1, len(df) + 1))
    return render_template(
        "index.html",
        dataset_options=dataset_options,
        **dataset_plots
    )


# =============================
# 4. AMBIL DATA + PREDIKSI BERDASARKAN INDEX
# =============================
@app.route("/get_dataset/<int:row_id>", methods=["GET"])
def get_dataset(row_id):
    idx = row_id - 1
    if idx < 0 or idx >= len(df):
        return jsonify({"error": "Index dataset tidak valid"})

    row = df.iloc[idx]
    data_dict = row.to_dict()

    df_input = pd.DataFrame([row[FEATURES]])
    df_input = pd.get_dummies(df_input)

    missing_cols = set(X.columns) - set(df_input.columns)
    for col in missing_cols:
        df_input[col] = 0

    extra_cols = set(df_input.columns) - set(X.columns)
    if extra_cols:
        df_input.drop(columns=list(extra_cols), inplace=True)

    df_input = df_input[X.columns]

    pred = int(model.predict(df_input)[0])
    proba = float(model.predict_proba(df_input)[0][1] * 100)

    data_dict["prediction"] = pred
    data_dict["prediction_label"] = "Positif Diabetes" if pred == 1 else "Negatif Diabetes"
    data_dict["prediction_probability"] = proba

    return jsonify(data_dict)


# =============================
# 5. PROSES PREDIKSI DARI FORM
# =============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {
            'gender': request.form['gender'],
            'age': float(request.form['age']),
            'hypertension': int(request.form['hypertension']),
            'heart_disease': int(request.form['heart_disease']),
            'smoking_history': request.form['smoking_history'],
            'bmi': float(request.form['bmi']),
            'HbA1c_level': float(request.form['HbA1c_level']),
            'blood_glucose_level': float(request.form['blood_glucose_level'])
        }

        df_input = pd.DataFrame([input_data])
        df_input = pd.get_dummies(df_input)

        missing_cols = set(X.columns) - set(df_input.columns)
        for col in missing_cols:
            df_input[col] = 0

        extra_cols = set(df_input.columns) - set(X.columns)
        if extra_cols:
            df_input.drop(columns=list(extra_cols), inplace=True)

        df_input = df_input[X.columns]

        prediction = int(model.predict(df_input)[0])
        probability = float(model.predict_proba(df_input)[0][1] * 100)

        result_text = "Positif Diabetes" if prediction == 1 else "Negatif Diabetes"
        color = "danger" if prediction == 1 else "success"

        # -----------------------------
        # BAR CHART -> BASE64 (TANPA STATIC)
        # -----------------------------
        plt.figure(figsize=(4, 3))
        plt.bar(['Tidak Diabetes', 'Diabetes'], [100 - probability, probability])
        plt.title("Probabilitas Diabetes (%)")
        plt.ylabel("Persentase")
        plt.tight_layout()
        bar_chart_base64 = fig_to_base64()

        # -----------------------------
        # GAUGE
        # -----------------------------
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            title={'text': "Risiko Diabetes (%)"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        gauge_json = json.dumps(gauge, cls=plotly.utils.PlotlyJSONEncoder)

        # -----------------------------
        # PIE CHART
        # -----------------------------
        pie = go.Figure(data=[go.Pie(
            labels=['Tidak Diabetes', 'Diabetes'],
            values=[100 - probability, probability],
            hole=0.4
        )])
        pie_json = json.dumps(pie, cls=plotly.utils.PlotlyJSONEncoder)

        dataset_options = list(range(1, len(df) + 1))

        return render_template(
            "index.html",
            prediction_text=result_text,
            probability_text=f"{probability:.2f}%",
            color=color,
            bar_chart_base64=bar_chart_base64,   # <--- base64
            gauge_json=gauge_json,
            pie_json=pie_json,
            dataset_options=dataset_options,
            **dataset_plots
        )

    except Exception as e:
        dataset_options = list(range(1, len(df) + 1))
        return render_template(
            "index.html",
            prediction_text=f"Terjadi kesalahan: {str(e)}",
            color="warning",
            dataset_options=dataset_options,
            **dataset_plots
        )


# =============================
# 6. HALAMAN EVALUASI MODEL
# =============================
@app.route("/evaluation")
def evaluation():

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    report_df = pd.DataFrame(
        classification_report(y, y_pred, output_dict=True)
    ).transpose()

    # -----------------------------
    # Confusion Matrix -> BASE64
    # -----------------------------
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_base64 = fig_to_base64()

    # -----------------------------
    # ROC-AUC -> BASE64
    # -----------------------------
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.tight_layout()
    roc_base64 = fig_to_base64()

    accuracy = accuracy_score(y, y_pred) * 100

    return render_template(
        "evaluation.html",
        accuracy=f"{accuracy:.2f}%",
        roc_auc=f"{roc_auc:.2f}",
        cm_base64=cm_base64,      # <--- base64
        roc_base64=roc_base64,    # <--- base64
        tables=[report_df.to_html(classes="table table-bordered text-center")]
    )


if __name__ == "__main__":
    app.run(debug=True)