import os
import sys
import pickle

from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

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

                report[model_name].append(test_model_score)

                # Log the models
                mlflow.sklearn.log_model(model, f"{model_name}_model")

                # Log the params and metrics
                # mlflow.log_params(gs.best_params_)
                mlflow.log_metrics(
                    {"train_model_score": train_model_score, "test_model_score": test_model_score})

        return report

    except Exception as e:
        raise CustomException(e, sys)
