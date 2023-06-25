import os
import sys
from dataclasses import dataclass
#sys.path.append('C:\\Users\\Pooja\\Downloads\\heart_mlops\\Stroke_prediction\src')
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC

from src.exceptions import CustomException
from src.logger import logging

from src.util import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest Classifier": RandomForestClassifier(),
                "KNearest Neighbor": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "XGBClassifier": XGBClassifier(), 
                #"CatBoosting Classifier": CatBoostClassifier(logging_level='Silent'),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "SVM" : SVC(),
            }
            params={
                "Logistic Regression": {
                    'solver':['newton-cg','lbfgs','liblinear','sag','saga'],
                    'penalty':['l2','elasticnet','none','l1']
                },

                "Random Forest Classifier": {
                #'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
               #'n_estimators': [8,16,32,64,128,256],
               'n_estimators': [8,16,32,64,128],
               #'max_features': ['auto', 'sqrt'],
               #'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
               'max_depth':[5,10,20,30],
               'min_samples_split': [2, 5, 10],
               #'min_samples_leaf': [1, 2, 4],
               #'bootstrap':[True, False]
                },

                "KNearest Neighbor": {
                    'n_neighbors': [i for i in range(1, 21, 2)],
                    'weights': ['uniform', 'distance'],
                    'metric':['euclidean', 'manhattan', 'minkowski']
                },

                "Decision Tree": {
                    'criterion':['gini','entropy'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "XGBClassifier": {
                   'max_depth':[i for i in range(3,10,2)],
                    #'max_depth': [3, 6, 10, 15],
                    #'n_estimators': [100, 250, 500],
                    #  'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4],
                    #  'subsample': np.arange(0.5, 1.0, 0.1),
                    #  'colsample_bytree': np.arange(0.5, 1.0, 0.1)
                },
               # "CatBoosting Classifier": {},
                "AdaBoost Classifier": {
                    'n_estimators':[5,10,50,100,1000]
                },
                "SVM": {
                'C': [0.1, 1, 10, 100, 1000], 
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
                },
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,param=params,
                                             models=models)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            print(best_model_name)

            predicted=best_model.predict(X_test)

            acc_score = accuracy_score(y_test, predicted)
            return acc_score
            



            
        except Exception as e:
            raise CustomException(e,sys)