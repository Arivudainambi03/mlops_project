import pandas as pd
from mlProject import logger
from mlProject.entity.config_entity import (DataValidationConfig)

class DataValidation:

    def __init__(self, config: DataValidationConfig) -> None:
        self.config = config

    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            all_columns = data.columns.to_list()
            all_schema = self.config.all_schema.keys()
            target_col = self.config.target_column.keys()

            for col in all_columns:
                if (col not in all_schema) and (col not in target_col):
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
        
            return validation_status
        
        except Exception as e:
            raise e
        