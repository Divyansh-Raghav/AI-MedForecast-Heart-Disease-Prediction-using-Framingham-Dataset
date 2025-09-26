

# 🩺 AI MedForecast – Heart Disease Prediction

A Machine Learning project that predicts the **10-year risk of Coronary Heart Disease (CHD)** using the **Framingham Heart Study dataset**.
The project leverages **Random Forest Classifier**, **SHAP explainability**, and an **interactive Gradio app** for real-time predictions.

---

## 🚀 Features

* ✅ Trained **Random Forest ML model** with **85%+ accuracy**.
* ✅ **Cross-validation** for performance robustness.
* ✅ **Feature importance analysis** and **SHAP-based explainability** for model transparency.
* ✅ **Interactive Gradio Web App** to predict heart disease risk based on patient inputs.
* ✅ Model persistence using **Joblib** for deployment readiness.

---

## 📂 Project Structure

```
AI-MedForecast/
│-- framingham_heart_study.csv      # Dataset (not included here, download separately)
│-- heart_disease_model.pkl         # Saved trained model
│-- app.py                          # Main project code
│-- README.md                       # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/your-username/AI-MedForecast.git
cd AI-MedForecast
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

> Example dependencies:

```txt
pandas
numpy
scikit-learn
matplotlib
shap
joblib
gradio
```

3. **Run the project**

```bash
python app.py
```

4. **Launch Gradio Web App** (uncomment `demo.launch()` in `app.py`)

```bash
python app.py
```

Access the app at: **[http://127.0.0.1:7860/](http://127.0.0.1:7860/)**

---

## 📊 Results

* **Accuracy**: ~85% (Random Forest)
* **Cross-validation score**: Stable across folds
* **Top features**: Age, Cholesterol, Systolic BP, BMI, Smoking status

---

## 🧠 Explainability

* **Feature Importance**: Identifies the top 10 clinical factors influencing predictions.
* **SHAP Values**: Provides transparent explanations of individual patient predictions.

---

## 🖥️ Example Usage

### Input:

* Age: 50
* Male: 0 (Female)
* Smoker: No
* Cholesterol: 245
* Systolic BP: 120
* BMI: 26

### Output:

```
Low Risk of CHD
```

---

## 📌 Future Improvements

* 🔹 Integrate with **Electronic Health Records (EHRs)**.
* 🔹 Add more ML models (XGBoost, Neural Networks) for comparison.
* 🔹 Deploy on **Streamlit/Flask** with cloud hosting.

---

## 🙌 Acknowledgements

* Dataset: [Framingham Heart Study](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)
* Libraries: Scikit-learn, SHAP, Gradio, Pandas, Matplotlib

---

Would you like me to also create a **`requirements.txt` file** (so that anyone cloning your repo can directly `pip install -r requirements.txt` and run)?
