import streamlit as st
import pandas as pd
import numpy as np
import pickle
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt


with open("model/model2.pkl", "rb") as f:
    ranclf = pickle.load(f)
with open("model/scaler2.pkl", "rb") as f:
    sc = pickle.load(f)
with open("model/minmax2.pkl", "rb") as f:
    mx = pickle.load(f)
with open("model/crop_dict2.pkl", "rb") as f:
    crop_dict, reverse_crop_dict = pickle.load(f)
x_train_orig = pd.read_csv("model/x_train_orig.csv")

def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    mx_features = mx.transform(features)
    sc_features = sc.transform(mx_features)
    prediction = ranclf.predict(sc_features)
    class_index = prediction.item()
    return reverse_crop_dict[class_index], class_index, features

def explain_lime_textually(exp, label_index, crop_name, top_n=3):
    st.markdown("#### LIME Explanation (Simple Terms)")
    lime_weights = exp.as_list(label=label_index)
    for feature, weight in lime_weights[:top_n]:
        direction = "positively influences" if weight > 0 else "negatively influences"
        st.write(f"- **{feature}**: This {direction} the recommendation for **{crop_name}** (impact = {weight:.4f})")


st.set_page_config(page_title=" Crop Recommender", layout="centered")
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1 {text-align: center; color: #2c3e50;}
    .stButton>button {background-color: #28a745; color: white; font-weight: bold;}
    .stForm input {background-color: #ecf0f1; border-radius: 5px;}
    .stAlert {border-radius: 10px; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)
st.title(" AI-Powered Crop Recommendation System")

with st.form("crop_form"):
    N = st.text_input("Nitrogen (N)")
    P = st.text_input("Phosphorus (P)")
    K = st.text_input("Potassium (K)")
    temperature = st.text_input("Temperature (°C)")
    humidity = st.text_input("Humidity (%)")
    ph = st.text_input("Soil pH")
    rainfall = st.text_input("Rainfall (mm)")
    submitted = st.form_submit_button("Predict Crop")

if submitted:
    try:
        
        Nf = float(N)
        Pf = float(P)
        Kf = float(K)
        Tf = float(temperature)
        Hf = float(humidity)
        Phf = float(ph)
        Rf = float(rainfall)
        
        result, class_index, features = recommendation(Nf, Pf, Kf, Tf, Hf, Phf, Rf)
        st.success(f" Recommended Crop: **{result}**")
        user_df = pd.DataFrame([{"N": Nf, "P": Pf, "K": Kf, "temperature": Tf,
                                 "humidity": Hf, "ph": Phf, "rainfall": Rf}])
        def scaled_predict_fn(x):
            return ranclf.predict_proba(sc.transform(mx.transform(x)))

        explainer = LimeTabularExplainer(
        training_data=x_train_orig.values, 
        feature_names=user_df.columns.tolist(),
        class_names=[reverse_crop_dict[i] for i in sorted(reverse_crop_dict)],
        mode='classification'
        )
        exp = explainer.explain_instance(user_df.iloc[0], predict_fn=scaled_predict_fn, labels=[class_index])
        st.subheader(" LIME Visual Explanation")
        fig = exp.as_pyplot_figure(label=class_index)
        st.pyplot(fig)

        
        explain_lime_textually(exp, class_index, result)

    except ValueError:
        st.error(" Please enter valid numeric values for all fields.")
