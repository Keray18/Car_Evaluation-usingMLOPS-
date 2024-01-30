from src.logger import logging
from src.exception import CustomException
from src.utils import eval_model, save_object
import os
import sys

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn
from urllib.parse import urlparse

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Beginning the model training phase...")
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1])

            # print(f"x_train: {x_train.shape}")
            # print(f"y_train: {y_train.shape}")
            models = {
                "SupportVectorClassifier": SVC(),
                "LogisticRegression": LogisticRegression(),
                "RandomForestClassifier": RandomForestClassifier(),
                "XGBClassifier": XGBClassifier(),
            }
            # params = {
            #     'SupportVectorClassifier': {
            #         'C': [0.1, 1, 10],
            #         'kernel': ['linear', 'rbf', 'poly']
            #     },
            #     'LogisticRegression': {
            #         'C': [0.1, 1, 10],
            #         'penalty': ['l1', 'l2']
            #     },
            #     'RandomForestClassifier': {
            #         'n_estimators': [50, 100, 200],
            #         'max_depth': [None, 10, 20]
            #     },
            #     'XGBClassifier': {
            #         'n_estimators': [50, 100, 200],
            #         'learning_rate': [0.01, 0.1, 0.2]
            #     },
            # }

            model_report: dict = eval_model(
                x_train, y_train, x_test, y_test, models)

            best_model_name = max(
                model_report, key=lambda k: max(model_report[k]))

            if max(model_report[best_model_name]) < 0.6:
                raise CustomException("No best model found.")

            best_model = models[best_model_name]
            logging.info("Best model found! Training was successful.")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
