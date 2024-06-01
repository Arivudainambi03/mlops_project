from typing import Any
from mlProject.config.configuration import ConfigurationManager
from mlProject.components.model_trainer import ModelTrainer
from mlProject import logger

STAGE_NAME = "Model Trainer stage"

class ModelTrainingTrainerPipeline:

    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config= config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()
        

if __name__ == '__main__':
    try:
        logger.info(f">>>> stage {STAGE_NAME} started. <<<<<")
        obj = ModelTrainingTrainerPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed. <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e