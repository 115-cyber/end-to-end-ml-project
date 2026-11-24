
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

data = CustomData(
    gender="male",
    race_ethnicity="group B",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="completed",
    reading_score=75,
    writing_score=80
)

input_df = data.get_data_as_data_frame()

print("Input Data:")
print(input_df)

pipeline = PredictPipeline()

prediction = pipeline.predict(input_df)

print("Predicted Mathematics Score:", prediction[0])
