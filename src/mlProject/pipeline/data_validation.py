from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_validation import DataValidation
from mlProject import logger


STAGE_NAME = "DATA VALIDATION STAGE"

class DataValidationTrainingPipeline:

    def __init__(self) -> None:
        pass

    def main(self):
        Config = ConfigurationManager()
        data_validation_config = Config.get_data_validation_config()
        data_validation = DataValidation(config= data_validation_config)
        data_validation.validate_all_columns()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started. <<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx===========x")
    
    except Exception as e:
        logger.exception(e)
        raise e