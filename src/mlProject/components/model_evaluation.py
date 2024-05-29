from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_score, recall_score
import pandas as pd
import joblib
import os
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
from mlProject.utils.common import save_json
from pathlib import Path


class ModelEvaluator:
    def _init_(self, models, params, scoring='accuracy', cv=5):
        self.models = models
        self.params = params
        self.scoring = scoring
        self.cv = cv

    def evaluate(self, train_x, train_y, test_x, test_y):
        best_models = {}
        best_scores = {}

        for model_name in self.models:
            print(f"Tuning hyperparameters for {model_name}...")
            grid_search = GridSearchCV(self.models[model_name], self.params[model_name], cv=self.cv, scoring=self.scoring)
            grid_search.fit(train_x, train_y)
            best_models[model_name] = grid_search.best_estimator_
            best_scores[model_name] = grid_search.best_score_
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"Best cross-validation score for {model_name}: {grid_search.best_score_}")

        best_model_name = max(best_scores, key=best_scores.get)
        best_model = best_models[best_model_name]   

        print(f"Best model: {best_model_name} with cross-validation score: {best_scores[best_model_name]}")

        # Evaluate the best model on the test set
        test_predictions = best_model.predict(test_x)
        classification_rep = classification_report(test_y, test_predictions)
        precision = precision_score(test_y, test_predictions, average='weighted')
        recall = recall_score(test_y, test_predictions, average='weighted')

        print(f"Classification Report for {best_model_name}:\n{classification_rep}")
        print(f"Precision for {best_model_name}: {precision}")
        print(f"Recall for {best_model_name}: {recall}")

        # return best_model_name, best_model, best_scores[best_model_name], classification_rep, precision, recall