import sys 
from dataclasses import dataclass

import numpy as np
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # encoding the labels and transforming values into one scale 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  # for handling missing values 
from sklearn.pipeline import Pipeline  # to create a pipeline 

from src.mlproject.exception import CustomException 
from src.mlproject.logger import logging
import os  # for saving pickle file