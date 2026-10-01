
import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)


@st.cache_resource
def load_model():
    model = joblib.load("crop_recommendation_ensemble_model.pkl")
    label_encoder = joblib.load("crop_recommendation_label_encoder.pkl")
    return model, label_encoder


model, label_encoder = load_model()


st.title("🌱 Crop Recommendation System")

st.markdown(
    """
    ### Smart Agriculture Using Machine Learning

    This application recommends the most suitable crop based on
    **soil nutrients and environmental conditions** using an
    **Ensemble Machine Learning Model**.
    """
)

st.divider()


st.subheader("🌾 Enter Soil & Environmental Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    nitrogen = st.number_input(
        "Nitrogen (N)",
        min_value=0.0,
        max_value=150.0,
        value=50.0,
        step=1.0
    )

    phosphorus = st.number_input(
        "Phosphorus (P)",
        min_value=0.0,
        max_value=150.0,
        value=50.0,
        step=1.0
    )

    potassium = st.number_input(
        "Potassium (K)",
        min_value=0.0,
        max_value=210.0,
        value=50.0,
        step=1.0
    )


with col2:
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=0.1
    )


with col3:
    ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5,
        step=0.1
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=100.0,
        step=1.0
    )


st.divider()


predict_button = st.button(
    "🔍 Recommend Crop",
    type="primary",
    use_container_width=True
)


if predict_button:

    input_data = pd.DataFrame(
        [[
            nitrogen,
            phosphorus,
            potassium,
            temperature,
            humidity,
            ph,
            rainfall
        ]],
        columns=[
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]
    )

    prediction = model.predict(input_data)

    predicted_class = prediction[0]

    predicted_crop = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    probabilities = model.predict_proba(input_data)[0]

    class_names = label_encoder.classes_

    probability_df = pd.DataFrame({
        "Crop": class_names,
        "Probability": probabilities
    })

    probability_df = probability_df.sort_values(
        by="Probability",
        ascending=False
    ).reset_index(drop=True)

    confidence = probability_df.iloc[0]["Probability"]


    st.divider()

    st.subheader("🌿 Crop Recommendation")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.success(
            f"### Recommended Crop: {predicted_crop.title()}"
        )

    with result_col2:

        st.info(
            f"### Confidence: {confidence * 100:.2f}%"
        )


    st.subheader("🏆 Top Crop Recommendations")

    top_3 = probability_df.head(3).copy()

    top_3["Probability"] = (
        top_3["Probability"] * 100
    ).round(2)

    top_3.columns = [
        "Crop",
        "Confidence (%)"
    ]

    top_3["Crop"] = top_3["Crop"].str.title()

    st.dataframe(
        top_3,
        use_container_width=True,
        hide_index=True
    )


    st.subheader("📊 Prediction Probabilities")

    chart_data = probability_df.copy()

    chart_data["Crop"] = chart_data["Crop"].str.title()

    chart_data = chart_data.set_index("Crop")

    st.bar_chart(
        chart_data["Probability"]
    )


    st.subheader("🧪 Input Conditions")

    input_display = pd.DataFrame({
        "Parameter": [
            "Nitrogen (N)",
            "Phosphorus (P)",
            "Potassium (K)",
            "Temperature",
            "Humidity",
            "Soil pH",
            "Rainfall"
        ],
        "Value": [
            nitrogen,
            phosphorus,
            potassium,
            temperature,
            humidity,
            ph,
            rainfall
        ],
        "Unit": [
            "kg/ha",
            "kg/ha",
            "kg/ha",
            "°C",
            "%",
            "pH",
            "mm"
        ]
    })

    st.dataframe(
        input_display,
        use_container_width=True,
        hide_index=True
    )


st.divider()

st.caption(
    "🌱 Crop Recommendation System | "
    "Powered by Ensemble Machine Learning"
)





