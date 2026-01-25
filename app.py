from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformation,DataTransformationConfig
from src.mlproject.components.model_trainer import ModelTrainer,ModelTrainerConfig
import sys
import warnings 
warnings.filterwarnings("ignore")

if __name__ == "__main__" :
    logging.info("The execution has started")

    try :
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path , test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation_config = DataTransformationConfig()
        data_tansformation = DataTransformation()
        train_arr,test_arr, _ = data_tansformation.initiate_data_transformation(train_data_path, test_data_path)

        model_trainer_config = ModelTrainerConfig()
        model_trainer = ModelTrainer()
        best_model_r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(best_model_r2_score)

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys) 
    
