import os 
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd 
from dotenv import load_dotenv  # For loading all the environment variables
import pymysql  # For connecting mysql db

import pickle 
import numpy as np 

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# Common Functionalities code are written here 

load_dotenv() # loads all the environment variables from .env file 
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")


def read_sql() :
    logging.info("Reading SQL Database Started")
    try :
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established", mydb)

        df = pd.read_sql_query("Select * from students", mydb)
        print(df.head())

        return df         
    except Exception as e :
        raise CustomException(e, sys)
    

def save_object(file_path, object):
    try :
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(object, file_obj)


    except Exception as e :
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try :
        report = {}

        for i in range(len(models)) :
            model = list(models.values())[i]
            parameter = params[list(models.keys())[i]]

            gs = GridSearchCV(model, parameter, cv=3)
            logging.info(f"Hyperparameter Tuning for {list(models.keys())[i]}")
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train) 

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            training_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        logging.info(f"Evaluation of Model is completed")
        return report 

    except Exception as e:
        raise CustomException(e, sys)