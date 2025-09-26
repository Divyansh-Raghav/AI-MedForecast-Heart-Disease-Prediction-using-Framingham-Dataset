# -------------------------------
# AI MedForecast - Minor Project
# Heart Disease Prediction (Framingham Dataset)
# -------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import shap
import matplotlib.pyplot as plt
import gradio as gr

# -------------------------------
# Load and preprocess dataset
# -------------------------------
df = pd.read_csv("framingham_heart_study.csv")

# Drop 'education' column and handle missing values
if "education" in df.columns:
    df = df.drop("education", axis=1)
df = df.fillna(df.median())

# Split into features (X) and target (y)
X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Random Forest model
# -------------------------------
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# -------------------------------
# Evaluate model
# -------------------------------
y_pred = rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring="accuracy")
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))

# -------------------------------
# Feature Importance
# -------------------------------
importances = rf_model.feature_importances_
features = X.columns
feat_importances = pd.Series(importances, index=features)
feat_importances.nlargest(10).plot(kind="barh")
plt.title("Top 10 Important Features (Random Forest)")
plt.show()

# -------------------------------
# SHAP Explainability
# -------------------------------
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Handle different SHAP output formats
if isinstance(shap_values, list):  
    shap_values_to_plot = shap_values[1]   # Class 1 (CHD = yes)
elif len(shap_values.shape) == 3:  
    shap_values_to_plot = shap_values[:, :, 1]
else:  
    shap_values_to_plot = shap_values

print(f"shap_values_to_plot shape: {shap_values_to_plot.shape}, X_test shape: {X_test.shape}")

# Plot summary
shap.summary_plot(shap_values_to_plot, X_test, feature_names=X_test.columns, show=True)

# -------------------------------
# Save model
# -------------------------------
joblib.dump(rf_model, "heart_disease_model.pkl")

# -------------------------------
# Predict for a new patient
# -------------------------------
new_patient_data = {
    "male": [0],
    "age": [50],
    "currentSmoker": [0],
    "cigsPerDay": [0.0],
    "BPMeds": [0.0],
    "prevalentStroke": [0],
    "prevalentHyp": [0],
    "diabetes": [0],
    "totChol": [245],
    "sysBP": [120],
    "diaBP": [80],
    "BMI": [26],
    "heartRate": [75],
    "glucose": [85],
}
new_patient_df = pd.DataFrame(new_patient_data)

# Ensure the order of columns is same as training
new_patient_df = new_patient_df[X_train.columns]

prediction = rf_model.predict(new_patient_df)[0]
print("New Patient Prediction:", "High Risk" if prediction == 1 else "Low Risk")

# -------------------------------
# Gradio Mini-App
# -------------------------------
def predict_heart_disease(
    male, age, currentSmoker, cigsPerDay, BPMeds,
    prevalentStroke, prevalentHyp, diabetes,
    totChol, sysBP, diaBP, BMI, heartRate, glucose
):
    input_data = pd.DataFrame(
        [[male, age, currentSmoker, cigsPerDay, BPMeds,
          prevalentStroke, prevalentHyp, diabetes,
          totChol, sysBP, diaBP, BMI, heartRate, glucose]],
        columns=X_train.columns
    )
    prediction = rf_model.predict(input_data)[0]
    return "High Risk of CHD" if prediction == 1 else "Low Risk of CHD"

demo = gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Number(label="Male (1=Male, 0=Female)"),
        gr.Number(label="Age"),
        gr.Number(label="Current Smoker (1=Yes, 0=No)"),
        gr.Number(label="Cigarettes Per Day"),
        gr.Number(label="BP Meds (1=Yes, 0=No)"),
        gr.Number(label="Prevalent Stroke (1=Yes, 0=No)"),
        gr.Number(label="Prevalent Hyp (1=Yes, 0=No)"),
        gr.Number(label="Diabetes (1=Yes, 0=No)"),
        gr.Number(label="Total Cholesterol"),
        gr.Number(label="Systolic Blood Pressure"),
        gr.Number(label="Diastolic Blood Pressure"),
        gr.Number(label="BMI"),
        gr.Number(label="Heart Rate"),
        gr.Number(label="Glucose"),
    ],
    outputs="text",
    title="AI MedForecast - Heart Disease Predictor",
    description="Enter patient details to predict 10-year risk of heart disease."
)

# Uncomment below line to launch Gradio UI
demo.launch()