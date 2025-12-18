import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        pass

    def start_train_pipeline(self):
        try:
            logging.info("Training pipeline started")
            
            # Data Ingestion
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed")
            
            # Data Transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)
            logging.info("Data transformation completed")
            
            # Model Training
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Model training completed with score: {model_score}")
            
            logging.info("Training pipeline completed successfully")
            
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.start_train_pipeline()