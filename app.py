import altair as alt
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)


FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

MODEL_NAMES = {
    "RandomForestClassifier": "Random Forest",
    "SVC": "Support Vector Machine",
    "KNeighborsClassifier": "K-Nearest Neighbours",
    "DecisionTreeClassifier": "Decision Tree",
}

# feature: (label, unit, min, max, step, display format, help text)
INPUTS = {
    "N": ("Nitrogen (N)", "kg/ha", 0.0, 150.0, 1.0, "%.0f",
          "Nitrogen content in the soil"),
    "P": ("Phosphorus (P)", "kg/ha", 0.0, 150.0, 1.0, "%.0f",
          "Phosphorus content in the soil"),
    "K": ("Potassium (K)", "kg/ha", 0.0, 210.0, 1.0, "%.0f",
          "Potassium content in the soil"),
    "ph": ("Soil pH", "pH", 0.0, 14.0, 0.1, "%.1f",
           "Acidity of the soil: below 7 is acidic, above 7 is alkaline"),
    "temperature": ("Temperature (°C)", "°C", -10.0, 60.0, 0.1, "%.1f",
                    "Average temperature during the growing season"),
    "humidity": ("Humidity (%)", "%", 0.0, 100.0, 0.1, "%.1f",
                 "Average relative humidity"),
    "rainfall": ("Rainfall (mm)", "mm", 0.0, 500.0, 1.0, "%.1f",
                 "Average rainfall"),
}

DEFAULTS = {
    "N": 50.0, "P": 50.0, "K": 50.0, "temperature": 25.0,
    "humidity": 70.0, "ph": 6.5, "rainfall": 100.0,
}

# Real samples from the unseen test set (notebook section 3.5)
EXAMPLES = {
    "rice": [60, 49, 44, 20.8, 84.5, 6.2, 240.1],
    "cotton": [140, 38, 15, 24.1, 75.9, 6.0, 69.9],
    "maize": [71, 54, 16, 22.6, 63.7, 5.7, 87.8],
    "chickpea": [25, 68, 77, 20.1, 15.1, 7.7, 85.7],
    "apple": [0, 133, 200, 23.7, 90.5, 5.7, 104.2],
    "coffee": [109, 31, 27, 23.1, 50.4, 7.0, 164.5],
}

CROPS = {
    "apple": ("Apple", "🍎"),
    "banana": ("Banana", "🍌"),
    "blackgram": ("Black Gram", "🫘"),
    "chickpea": ("Chickpea", "🫘"),
    "coconut": ("Coconut", "🥥"),
    "coffee": ("Coffee", "☕"),
    "cotton": ("Cotton", "☁️"),
    "grapes": ("Grapes", "🍇"),
    "jute": ("Jute", "🌿"),
    "kidneybeans": ("Kidney Beans", "🫘"),
    "lentil": ("Lentil", "🫘"),
    "maize": ("Maize", "🌽"),
    "mango": ("Mango", "🥭"),
    "mothbeans": ("Moth Beans", "🫘"),
    "mungbean": ("Mung Bean", "🫛"),
    "muskmelon": ("Muskmelon", "🍈"),
    "orange": ("Orange", "🍊"),
    "papaya": ("Papaya", "🍈"),
    "pigeonpeas": ("Pigeon Peas", "🫛"),
    "pomegranate": ("Pomegranate", "🍎"),
    "rice": ("Rice", "🌾"),
    "watermelon": ("Watermelon", "🍉"),
}


def crop_name(crop):
    return CROPS.get(crop, (crop.title(), "🌱"))[0]


def crop_emoji(crop):
    return CROPS.get(crop, (crop.title(), "🌱"))[1]


def confidence_level(confidence):
    if confidence >= 0.8:
        return "High"
    if confidence >= 0.5:
        return "Medium"
    return "Low"


st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 55%, #66bb6a 100%);
        color: #ffffff;
        padding: 1.75rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.25rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        margin-top: 0.4rem;
        opacity: 0.92;
    }
    .result-card {
        display: flex;
        align-items: center;
        gap: 1.25rem;
        background: linear-gradient(135deg, #1b5e20 0%, #43a047 100%);
        color: #ffffff;
        padding: 1.5rem 1.75rem;
        border-radius: 16px;
        margin-bottom: 1rem;
    }
    .result-emoji {
        font-size: 3.5rem;
        line-height: 1;
    }
    .result-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.85;
    }
    .result-crop {
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .result-pill {
        display: inline-block;
        margin-top: 0.4rem;
        padding: 0.2rem 0.75rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.2);
        font-size: 0.9rem;
        font-weight: 600;
    }
    .rank-card {
        text-align: center;
    }
    .rank-emoji {
        font-size: 2rem;
    }
    .rank-crop {
        font-weight: 600;
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Artifacts saved by Smart_Crop_Recommendation.ipynb (section 2.5)
@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    cap_bounds = joblib.load("cap_bounds.pkl")
    return model, scaler, label_encoder, cap_bounds


model, scaler, label_encoder, cap_bounds = load_model()

model_name = MODEL_NAMES.get(type(model).__name__, type(model).__name__)


def preprocess(raw_df):
    # Same steps as training: IQR capping, then standard scaling
    capped = raw_df.copy()
    for feature in FEATURES:
        low, high = cap_bounds[feature]
        capped[feature] = capped[feature].clip(low, high)
    return scaler.transform(capped[FEATURES])


for feature in FEATURES:
    st.session_state.setdefault(feature, DEFAULTS[feature])


def load_example():
    example = st.session_state.example
    if example in EXAMPLES:
        values = dict(zip(FEATURES, EXAMPLES[example]))
    else:
        values = DEFAULTS
    for feature in FEATURES:
        st.session_state[feature] = float(values[feature])


def number_input(feature):
    label, _, min_value, max_value, step, display_format, help_text = INPUTS[feature]
    return st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        step=step,
        format=display_format,
        help=help_text,
        key=feature
    )


with st.sidebar:
    st.header("🌱 About")
    st.write(
        "This app recommends the crop best suited to your soil and "
        "climate, using a machine learning model trained on the "
        "Kaggle Crop Recommendation dataset."
    )

    st.subheader("How it works")
    st.markdown(
        """
        1. Enter your soil nutrients and climate conditions
        2. Click **Recommend Crop**
        3. Review the best crop and the alternatives
        """
    )

    st.subheader("Model")
    st.markdown(
        f"""
        - **Algorithm:** {model_name}
        - **Crops:** {len(label_encoder.classes_)}
        - **Inputs:** 7 soil & climate features
        - **Preprocessing:** IQR capping + standard scaling
        """
    )

    st.markdown(
        "[Dataset on Kaggle]"
        "(https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)"
    )


st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">🌱 Crop Recommendation System</div>
        <div class="hero-subtitle">
            Find the most suitable crop for your field from soil nutrients
            and climate conditions, powered by <b>{model_name}</b>.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


input_col, result_col = st.columns([5, 7], gap="large")

with input_col:

    st.subheader("🌾 Field Conditions")

    st.selectbox(
        "Try an example from the test set",
        ["custom"] + list(EXAMPLES),
        format_func=lambda crop: (
            "✏️ Custom values" if crop == "custom"
            else f"{crop_emoji(crop)} {crop_name(crop)} field"
        ),
        key="example",
        on_change=load_example
    )

    with st.form("conditions", border=False):

        with st.container(border=True):
            st.markdown("**🧪 Soil**")
            soil_col1, soil_col2 = st.columns(2)
            with soil_col1:
                number_input("N")
                number_input("K")
            with soil_col2:
                number_input("P")
                number_input("ph")

        with st.container(border=True):
            st.markdown("**🌦️ Climate**")
            climate_col1, climate_col2 = st.columns(2)
            with climate_col1:
                number_input("temperature")
                number_input("rainfall")
            with climate_col2:
                number_input("humidity")

        submitted = st.form_submit_button(
            "🔍 Recommend Crop",
            type="primary",
            width="stretch"
        )


with result_col:

    st.subheader("🌿 Recommendation")

    if not submitted:

        st.info(
            "Enter your field conditions on the left, or pick an example, "
            "then click **Recommend Crop** to see the best crop for your field."
        )

    else:

        values = {feature: st.session_state[feature] for feature in FEATURES}

        input_data = pd.DataFrame([values], columns=FEATURES)

        model_input = preprocess(input_data)

        predicted_class = model.predict(model_input)[0]

        predicted_crop = label_encoder.inverse_transform(
            [predicted_class]
        )[0]

        probabilities = model.predict_proba(model_input)[0]

        probability_df = pd.DataFrame({
            "crop": label_encoder.classes_,
            "Probability": probabilities
        }).sort_values(
            by="Probability",
            ascending=False
        ).reset_index(drop=True)

        probability_df["Crop"] = probability_df["crop"].map(crop_name)

        confidence = probability_df.iloc[0]["Probability"]

        level = confidence_level(confidence)

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-emoji">{crop_emoji(predicted_crop)}</div>
                <div>
                    <div class="result-label">Recommended crop</div>
                    <div class="result-crop">{crop_name(predicted_crop)}</div>
                    <span class="result-pill">
                        {level} confidence · {confidence * 100:.1f}%
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if level == "Low":
            st.warning(
                "**Low confidence.** These conditions don't closely match "
                "any crop in the training data. Double-check your values "
                "and consider the alternatives below."
            )

        st.markdown("**🏆 Top 3 crops**")

        medals = ["🥇", "🥈", "🥉"]

        for rank_col, medal, (_, row) in zip(
            st.columns(3), medals, probability_df.head(3).iterrows()
        ):
            with rank_col:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="rank-card">
                            <div class="rank-emoji">{medal} {crop_emoji(row["crop"])}</div>
                            <div class="rank-crop">{row["Crop"]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.progress(
                        float(row["Probability"]),
                        text=f"{row['Probability'] * 100:.1f}%"
                    )

        chart_tab, input_tab = st.tabs(
            ["📊 Probabilities", "🧾 Your Inputs"]
        )

        with chart_tab:

            chart_data = probability_df[
                probability_df["Probability"] > 0
            ].head(10).copy()

            chart_data["Type"] = [
                "Recommended" if i == 0 else "Other"
                for i in range(len(chart_data))
            ]

            chart = alt.Chart(chart_data).mark_bar(
                cornerRadiusEnd=4
            ).encode(
                x=alt.X(
                    "Probability:Q",
                    title="Probability",
                    axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1])
                ),
                y=alt.Y("Crop:N", sort="-x", title=None),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(
                        domain=["Recommended", "Other"],
                        range=["#2e7d32", "#a5d6a7"]
                    ),
                    legend=None
                ),
                tooltip=[
                    alt.Tooltip("Crop:N"),
                    alt.Tooltip("Probability:Q", format=".1%")
                ]
            ).properties(
                height=max(120, 34 * len(chart_data))
            )

            st.altair_chart(chart)

            st.caption("Crops with a probability above 0%, up to the top 10.")

        with input_tab:

            input_display = pd.DataFrame({
                "Parameter": [INPUTS[feature][0] for feature in FEATURES],
                "Value": [values[feature] for feature in FEATURES],
                "Unit": [INPUTS[feature][1] for feature in FEATURES],
            })

            st.dataframe(
                input_display,
                width="stretch",
                hide_index=True
            )


st.divider()

st.caption(
    "🌱 Crop Recommendation System · "
    f"Powered by {model_name} · "
    "Walchand College of Engineering, Sangli"
)
