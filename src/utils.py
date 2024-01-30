import os
import sys
import pickle

from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def eval_model(x_train, y_train, x_test, y_test, models):
    try:
        report = {key: [] for key in models.keys()}

        for model_name, model in models.items():
            # para = params[model_name]

            with mlflow.start_run():
                # gs = GridSearchCV(model, para, cv=3)
                # gs.fit(x_train, y_train)

                # model.set_params(**gs.best_params_)
                model.fit(x_train, y_train)

                y_train_pred = model.predict(x_train)
                y_test_pred = model.predict(x_test)

                train_signature = infer_signature(x_train, y_train_pred)
                test_signature = infer_signature(x_test, y_test_pred)

                train_model_score = accuracy_score(y_train, y_train_pred)
                test_model_score = accuracy_score(y_test, y_test_pred)
                train_precision = precision_score(
                    y_train, y_train_pred, average='weighted')
                test_precision = precision_score(
                    y_test, y_test_pred, average='weighted')

                train_recall = recall_score(
                    y_train, y_train_pred, average='weighted')
                test_recall = recall_score(
                    y_test, y_test_pred, average='weighted')

                train_f1 = f1_score(y_train, y_train_pred, average='weighted')
                test_f1 = f1_score(y_test, y_test_pred, average='weighted')

                report[model_name]['accuracy'].append(
                    {'train': train_accuracy, 'test': test_accuracy})
                report[model_name]['precision'].append(
                    {'train': train_precision, 'test': test_precision})
                report[model_name]['recall'].append(
                    {'train': train_recall, 'test': test_recall})
                report[model_name]['f1'].append(
                    {'train': train_f1, 'test': test_f1})

                # Log the models
                mlflow.sklearn.log_model(model, f"{model_name}_model")

                # Log the params and metrics
                # mlflow.log_params(gs.best_params_)
                mlflow.log_metrics({
                    "train_accuracy": train_accuracy, "test_accuracy": test_accuracy,
                    "train_precision": train_precision, "test_precision": test_precision,
                    "train_recall": train_recall, "test_recall": test_recall,
                    "train_f1": train_f1, "test_f1": test_f1
                })

        return report

    except Exception as e:
        raise CustomException(e, sys)
