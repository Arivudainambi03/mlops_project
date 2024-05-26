import os
import json
import pandas as pd
from mlProject import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from mlProject.entity.config_entity import (DataTransformationConfig)


class DataTransforamtion:

    def __init__(self, config:DataTransformationConfig) -> None:
        self.config = config

    def rename_columns(self, df, dct:dict):
        return df.rename(columns = dct)

    def repalce_values_column(self, df, column_name, old_value, new_value):
        df[column_name] = df[column_name].replace(old_value, new_value)
        return df

    def Recuresive_Feature_Elimination(self, X, y, no_features_to_select)->list:
        model = RandomForestClassifier()
        rfe = RFE(model, no_features_to_select = no_features_to_select)
        fit = rfe.fit(X, y)
        selected_features = X.columns[fit.support_]
        return selected_features.to_list()

    def convert_categorical_to_numerical(self, df, output_file='category_mappings.json'):

        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        # Initialize dictionary to store mappings
        category_mappings = {}

        # Convert categorical columns to numerical
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            # Convert numpy int32 to Python int
            category_mappings[col] = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}

        # Save mappings to a JSON file
        with open(os.join(self.config.root_dir, output_file), 'w') as file:
            json.dump(category_mappings, file, indent=4)

        return df, category_mappings

    def train_test_splitting(self):

        data = pd.read(self.config.data_path)

        # split the dataframe
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index = False)

        logger.info("Splitting the dataset into train and test dataset.")
        logger.info(f"Training dataset shape is {train.shape}.")
        logger.info(f"Test dataset shape id {test.shape}.")

        print(train.shape,  test.shape)
