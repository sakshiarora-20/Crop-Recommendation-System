Got it. I’ll adjust the README to mention Streamlit explicitly and remove Flask references.

---

# Crop Recommendation System (CRS + XAI)

A machine learning based Streamlit application that recommends suitable crops using soil nutrient and weather parameters. The system integrates Explainable AI (SHAP and LIME) so that users can understand why a particular crop is recommended.

---

## Features

* Crop recommendation using ML models
* Streamlit based user interface
* XAI based explanation using SHAP and LIME
* Accepts soil and climate inputs

---

## Models Used

* Random Forest (best performing)
* XGBoost
* Logistic Regression
* Decision Tree
* SVM
* KNN

---

## Input Parameters

N, P, K, Temperature, Humidity, pH, Rainfall

---

## Dataset

Crop Recommendation Dataset from Kaggle
[https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

---

## CRS + XAI

This project combines machine learning with Explainable Artificial Intelligence:

* SHAP explains global feature contribution
* LIME explains each individual prediction

This makes recommendations interpretable and useful in real agricultural decision making.

---

## Tech Stack

Python, Streamlit, Scikit-Learn, Pandas, NumPy, SHAP, LIME

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
