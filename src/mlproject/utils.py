import os 
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import pandas as pd 
from dotenv import load_dotenv  # For loading all the environment variables
import pymysql  # For connecting mysql db

import pickle 
import numpy as np 

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