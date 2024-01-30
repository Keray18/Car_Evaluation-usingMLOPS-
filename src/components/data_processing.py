import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataProcessingConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataProcessing:
    def __init__(self):
        self.data_processing_config = DataProcessingConfig()

    def get_data_processing_object(self):
        try:
            logging.info('Creating data processing object...')
            columns = [
                "buying",
                "maint",
                "doors",
                "persons",
                "lug_boot",
                "safety"
            ]
            pipeline = Pipeline(steps=[
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer(
                [
                    ("pipeline", pipeline, columns)
                ]
            )
            logging.info(f"preprocessing object has been created.")
            return preprocessor

        except Exception as e:
            raise CustomException(sys, e)

    def initiate_data_processing(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading the train and test dataset.")

            preprocessing_obj = self.get_data_processing_object()
            target_col = "target"

            input_features_train_df = train_df.drop(
                columns=[target_col], axis=1)
            target_feature_train_df = train_df[target_col]
            input_feature_test_df = test_df.drop(
                columns=[target_col], axis=1)
            target_feature_test_df = test_df[target_col]

            logging.info("Preprocessing the datasets.")
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)

            target_encoder = OneHotEncoder()
            target_feature_train_arr = target_encoder.fit_transform(
                target_feature_train_df.values.reshape(-1, 1))
            target_feature_test_arr = target_encoder.transform(
                target_feature_test_df.values.reshape(-1, 1))

            # print(f"input_feature_train_arr: {input_feature_train_arr.shape}")
            # print(
            #     f"target_feature_train_arr: {target_feature_train_arr.shape}")

            train_arr = np.c_[input_feature_train_arr,
                              target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]
            save_object(
                file_path=self.data_processing_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Data Preprocessing has been successfully completed.")
            return (
                train_arr,
                test_arr,
                self.data_processing_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(sys, e)
