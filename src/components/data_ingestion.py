import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            car_evaluation = fetch_openml(name='car', version=3)
            df = pd.DataFrame(data=car_evaluation.data,
                              columns=car_evaluation.feature_names)
            df['target'] = car_evaluation.target
            logging.info("Read the dataset.")
            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,
                      index=False, header=True)
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42)
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,
                            index=False, header=True)

            logging.info("Data Ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path


            )

        except Exception as e:
            raise CustomException(sys, e)
