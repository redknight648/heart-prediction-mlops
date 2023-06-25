import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
from sklearn.impute import KNNImputer
from src.exceptions import CustomException
from src.logger import logging
import os

from src.util import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["age", "hypertension","heart_disease","avg_glucose_level","bmi"]
            categorical_columns = [
                "gender",
                "ever_married",
                "work_type",
                "Residence_type",
                "smoking_status",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",KNNImputer(n_neighbors=4, weights="uniform")),

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                # ("imputer",KNNImputer(n_neighbors=4, weights="uniform")),
                ("one_hot_encoder",OneHotEncoder()),
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="stroke"
            remove_col = 'id'
            numerical_columns = ["age", "hypertension","heart_disease","avg_glucose_level","bmi"]

            # input_feature_train_df=train_df.drop(columns=[target_column_name,remove_col],axis=1)
            target_feature_train_df=train_df[target_column_name]

            # input_feature_test_df=test_df.drop(columns=[target_column_name,remove_col],axis=1)
            target_feature_test_df=test_df[target_column_name]
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            input_feature_train_df = input_feature_train_df.iloc[:, 1:]  # Remove extra column at index 0

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            input_feature_test_df = input_feature_test_df.iloc[:, 1:]



            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)