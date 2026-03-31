#  Stroke Prediction using Machine Learning Classification Algorithm

A machine-learning project focused on predicting the likelihood of a stroke using classification models and patient demographic, lifestyle, and medical data.

---

##  Project Overview

This project builds an end-to-end pipeline: loading patient data (age, gender, hypertension, heart disease, smoking status, BMI, etc.), performing exploratory analysis, engineering features, training classification models (e.g., Logistic Regression, Random Forest, XGBoost), evaluating their performance, and deriving actionable insights for health interventions. The goal is to classify patients into high vs low risk of stroke and support early preventive care.

---

##  Tech Stack

* **Language:** Python
* **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
* **Environment:** Jupyter Notebook / Google Colab

---

##  Workflow Summary

### 1. Data Collection

Dataset includes features such as: age, gender, hypertension, heart disease, ever married, work type, residence type, average glucose level, BMI, smoking status, and target variable “stroke” (0/1 or yes/no).

### 2. Exploratory Data Analysis (EDA)

* Distribution of stroke vs non-stroke patients by age, BMI, glucose level
* Boxplots and histograms for key features grouped by target
* Correlation matrix among numeric features and relation to stroke
* Check missing values, class imbalance (strokes are rarer)

### 3. Feature Engineering

* Encode categorical features (gender, work type, smoking status) via one-hot encoding or label encoding
* Create derived features such as BMI category, glucose level buckets, combined risk score (e.g., hypertension + heart disease)
* Scale numeric features if required
* Handle class imbalance via oversampling, undersampling, or class weights
* Split data into training and test sets (e.g., 80/20) with stratification

### 4. Modeling

Classification algorithms used:

* **Logistic Regression** (baseline)
* **Random Forest Classifier** (strong performer)
* **XGBoost / LightGBM** for advanced performance
  Hyper-parameter tuning via GridSearchCV/RandomizedSearchCV (e.g., n_estimators, max_depth, learning_rate)

### 5. Evaluation

Metrics used:

* Accuracy
* Precision, Recall, F1-Score
* Confusion Matrix
* ROC-AUC
  **Result:** The best model achieved high recall and acceptable precision, enabling identification of patients at risk of stroke for early intervention.

### 6. Insights & Application

* Key risk-factors: age, average glucose level, BMI, hypertension, heart disease emerged as strong predictors
* Practical recommendations: healthcare providers may prioritise screening for patients with elevated glucose, higher BMI, and combined hypertension + heart-disease history
* Model supports early risk stratification and targeted patient monitoring

---

##  Project Structure

```
Stroke-Prediction-ML-Classification/
│── data/
│   ├── raw/
│   └── processed/
│── notebooks/
│   └── stroke_prediction_analysis.ipynb
│── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── evaluate.py
│── README.md
│── requirements.txt
```

---

## 📈 Key Findings

* Patients with hypertension and heart disease together had markedly higher predicted risk of stroke.
* Deriving combined risk features (e.g., hypertension + heart-disease flag) improved model discrimination.
* Tree-based classifiers like Random Forest and XGBoost outperformed Logistic Regression due to capturing complex interactions.
* Managing class imbalance (e.g., via SMOTE or class weighting) was critical to achieve strong recall for the minority ‘stroke’ class.

---

## 🚀 Future Improvements

* Incorporate longitudinal data (e.g., yearly health records, lifestyle change logs) to improve prediction of future strokes.
* Deploy as a clinical web/app interface where patient data can be input and risk score returned in real time.
* Include interpretability/explainability (e.g., SHAP values) so clinicians understand which factors drove the prediction.
* Monitor model performance over time in production, adjust for population shift or new risk-factors.
* Link to intervention module: when high risk flagged, automatic referral to specialist, lifestyle recommendation or alert system.

---
