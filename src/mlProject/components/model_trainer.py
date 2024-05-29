import pandas as pd
import os
from mlProject import logger
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump


class ModelTrainer:
    
    def __init__(self, config:ModelTrainerConfig, modelconfig, paramsconfig) -> None:
        self.config = config
        self.model_config = modelconfig
        self.params_config = paramsconfig

    def train(self):
        """training the funtion 
        Model training and the parameter tuning both the operation will be performed 

        return:
            best model will be exported.
        
        """
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_columns], axis= 1)
        test_x = train_data.drop([self.config.target_columns], axis = 1)
        train_y = train_data[[self.config.target_columns]]
        test_y = test_data[[self.config.target_columns]]

        # Create models dictionary from config
        models = {name: eval(self.model_config['models'][name]['class'])() for name in self.model_config['models']}

        # Extract parameters from config
        params = self.params_config['params']

        best_models = {}
        best_scores = {}

        # Assuming X_train, y_train, X_test, and y_test are predefined
        for model_name in models:
            print(f"Tuning hyperparameters for {model_name}...")
            grid_search = GridSearchCV(models[model_name], params[model_name], cv=5, scoring='accuracy')
            grid_search.fit(train_x, train_y)
            best_models[model_name] = grid_search.best_estimator_
            best_scores[model_name] = grid_search.best_score_
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"Best cross-validation score for {model_name}: {grid_search.best_score_}")

        best_model_name = max(best_scores, key=best_scores.get)
        best_model = best_models[best_model_name]

        print(f"Best model: {best_model_name} with cross-validation score: {best_scores[best_model_name]}")
        
        joblib.dump(best_model_name, os.path.join(self.config.root_dir, self.config.model_name))