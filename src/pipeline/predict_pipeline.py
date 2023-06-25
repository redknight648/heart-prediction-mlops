import sys
import os
import pandas as pd
from src.exceptions import CustomException
from src.util import load_object


class PredictPipeline:
    def _init_(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        age: float,
        heart_disease: int,
        hypertension: int,
        ever_married: str,
        work_type: str,
        Residence_type: str,
        avg_glucose_level: float,
        bmi: float,
        smoking_status: str):

        self.gender = gender

        self.age = age

        self.heart_disease = heart_disease
        self.hypertension = hypertension

        self.ever_married = ever_married

        self.work_type = work_type

        self.Residence_type = Residence_type
        self.avg_glucose_level = avg_glucose_level
        self.bmi =bmi
        self.smoking_status =smoking_status

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "age": [self.age],
                "heart_disease": [self.heart_disease],
                "hypertension": [self.hypertension],
                "ever_married": [self.ever_married],
                "work_type": [self.work_type],
                "Residence_type": [self.Residence_type],
                "avg_glucose_level": [self.avg_glucose_level],
                "bmi": [self.bmi],
                "smoking_status": [self.smoking_status]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)