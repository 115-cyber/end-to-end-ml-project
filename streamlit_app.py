
import streamlit as st
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("Student Performance Predictor")
st.write("Enter the student details to predict their mathematics score.")

# Initialize prediction pipeline
@st.cache_resource
def load_pipeline():
    return PredictPipeline()

predict_pipeline = load_pipeline()

# Input form
with st.form("prediction_form"):

    st.subheader("Student Information")

    gender = st.selectbox(
        "Gender",
        options=["male", "female"]
    )

    ethnicity = st.selectbox(
        "Race / Ethnicity",
        options=["group A", "group B", "group C", "group D", "group E"]
    )

    parental_level_of_education = st.selectbox(
        "Parental Level of Education",
        options=[
            "associate's degree",
            "bachelor's degree",
            "high school",
            "master's degree",
            "some college",
            "some high school"
        ]
    )

    lunch = st.selectbox(
        "Lunch",
        options=["standard", "free/reduced"]
    )

    test_preparation_course = st.selectbox(
        "Test Preparation Course",
        options=["none", "completed"]
    )

    st.subheader("Academic Scores")

    reading_score = st.number_input(
        "Reading Score",
        min_value=0,
        max_value=100,
        value=70
    )

    writing_score = st.number_input(
        "Writing Score",
        min_value=0,
        max_value=100,
        value=70
    )

    submit = st.form_submit_button("Predict Mathematics Score")

# Prediction
if submit:

    try:
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        pred_df = data.get_data_as_data_frame()

        st.write("### Input Data")
        st.dataframe(pred_df)

        results = predict_pipeline.predict(pred_df)

        st.success("Prediction completed successfully!")

        st.metric(
            label="Predicted Mathematics Score",
            value=f"{results[0]:.2f}"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
