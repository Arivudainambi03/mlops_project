import os
import urllib.request as request
import zipfile
from mlProject import logger
from mlProject.utils.common import get_size
from pathlib import Path
from mlProject.entity.config_entity import (DataIngestingConfig)


class DataIngestion:

    def __init__(self, config: DataIngestingConfig) -> None:
        self.config = config

    def downloaded_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_url,
                filename= self.config.local_data_file
            )

            logger.info(f"{filename} downloade ! with following {headers} names.")
        
        else:
            logger.info(f"File already exists of size of {get_size(Path(self.config.local_data_file))}")
    
    def extract_zip_file(self):
        """
        zip_file_path : str
        Extract the zip file to the directory

        Returns:
            None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

