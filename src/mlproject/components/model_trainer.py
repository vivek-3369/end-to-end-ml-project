import os 
import sys 
from dataclasses import dataclass 
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object,evaluate_models

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer :
    def __init__(self) :
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(f"Split training and testing input data")
            X_train,y_train,X_test,y_test=(
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:,:-1],
                test_array[:, -1]
            )

            # Creating a model list
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "XGBoost Regressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
            }

            # Hyperparameter Tuning
            params={
                "Linear Regression":{},
                "Decision Tree Regressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "CatBoost Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "XGBoost Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting Regressor":{
                    #'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    #'criterion':['squared_error', 'friedman_mse'],
                    #'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Random Forest Regressor":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                }      
            }

            model_report:dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # To get the best model score from the dict 
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from the dict 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]


            # Raising an custom exception if best model acc lesser than 0.6
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info("Best model found on both training and test datasets")

            # Saving the best model in a pickle file 
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                object = best_model
            )

            prediction = best_model.predict(X_test)

            final_r2_score = r2_score(y_test, prediction)

            return final_r2_score 

        except Exception as e :
            raise CustomException(e, sys)