from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_ingestion import DataIngestion
from mlProject import logger

STAGE_NAME = "Data Ingetsion Stage"

class DataIngestionTraningPipeline:

    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_cofig = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_cofig)
        data_ingestion.downloaded_file()
        data_ingestion.extract_zip_file()

    
if __name__ == '__main__':
    try:
        logger.info(f">>>> stage {STAGE_NAME} started.")
        obj = DataIngestionTraningPipeline()
        obj.main()
        logger.info(f">>> stage {STAGE_NAME} commpleted. <<<<\n\nx=============x")
    except Exception as e:
        logger.exception(e)
        raise e