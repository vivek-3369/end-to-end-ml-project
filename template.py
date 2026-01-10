import os 
from pathlib import Path  # for creating paths 
import logging

# For creating the entire project structure

logging.basicConfig(level=logging.INFO)

project_name = "mlproject"

list_of_file = [
    f"src/{project_name}/__init__.py", 
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitoring.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_file :
    filepath = Path(filepath)
    file_dir, file_name = os.path.split(filepath)

    if file_dir != "" :
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for the {file_name}")
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f :
            pass 
            logging.info(f"Creating an empty file: {filepath}")

    else :
        logging.info(f"{file_name} is already exists")