from typing import Any
from mlProject.constants import *
from mlProject.utils.common import create_directories , read_yaml
from mlProject.entity.config_entity import  (DataIngestingConfig,
                                             DataValidationConfig,
                                             DataTransformationConfig,
                                             ModelTrainerConfig)


class ConfigurationManager:

    def __init__(self,
                 config_file_path = CONFIG_FILE_PATH,
                 params_file_path = PARAMS_FILE_PATH,
                 schema_file_path = SCHEMA_FILE_PATH,
                 model_file_path = MODEL_FILE_PATH):
        
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        self.schema = read_yaml(schema_file_path)
        self.model = read_yaml(model_file_path)

        create_directories(filepath = [self.config.artifacts_root])


    def get_data_ingestion_config(self)-> DataIngestingConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestingConfig(
                                    root_dir=config.root_dir,
                                    source_url=config.source_url,
                                    local_data_file = config.local_data_file,
                                    unzip_dir=config.unzip_dir)
        
        return data_ingestion_config
    
    def get_data_validation_config(self)-> DataValidationConfig:
        config = self.config.data_validatation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
                                    root_dir=config.root_dir,
                                    STATUS_FILE = config.STATUS_FILE,
                                    unzip_data_dir = config.unzip_data_dir, 
                                    all_schema = schema
                                    )
        
        return data_validation_config
    
    def get_data_transformation_config(self)-> DataTransformationConfig:
        config = self.config.data_Transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
                                        root_dir = config.root_dir,
                                        data_path = config.data_path
                                        )

        return data_transformation_config

    def get_model_trainer_config(self)-> ModelTrainerConfig:
        config = self.config.model_trainer

        create_directories([config.root_dir])

        model_trainer = ModelTrainerConfig( 
                                root_dir = config.root_dir,
                                train_data_path = config.train_data_path,
                                test_data_path = config.test_data_path,
                                model_name = config.model_name,
                                target_columns= self.schema.TARGET_COLUMN,
                                model_dir = config.root_dir
                                )

        return model_trainer
    
    