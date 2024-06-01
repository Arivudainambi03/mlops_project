from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_transformation import DataTransformation
from mlProject import logger
from pathlib import Path
import pandas as pd

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingConfig:

    def __init__(self) -> None:
        self.dataset = pd.read_csv(r'artifacts\data_ingestion\bank-additional-full.csv')

    def Col_Transformation(self, df: pd.DataFrame, data_transformation):

        try:
            col = {'emp.var.rate' : 'emp_var_rate',
                    'cons.price.idx' : 'cons_price_idx',
                    'cons.conf.idx': 'cons_conf_idx',
                    'nr.employed' :'nr_employed'}

            df = data_transformation.rename_columns(df, dct = col)
            df = data_transformation.repalce_values_column(df, 'education', "university.degree" , "university_degree")
            df = data_transformation.repalce_values_column(df, 'education', "high.school", "high_school")
            df = data_transformation.repalce_values_column(df, 'education', "basic.9y", "basic_9y")
            df = data_transformation.repalce_values_column(df, 'education', "professional.course", "professional_course")
            df = data_transformation.repalce_values_column(df, 'education', "basic.4y" , "basic_4y")
            df = data_transformation.repalce_values_column(df, 'education', "basic.6y", "basic_6y")
            
            logger.info(f"Columns Transformation are done.")
            return df
        
        except Exception as e:
            raise e 
        
    def main(self):
        try:
            with open(Path("artifacts\data_validation\status.txt"), 'r') as f:
                status = f.read().split(" ")[-1]
                print(status)
                
            if status == "True":
                
                config = ConfigurationManager()
                DataTransforamtion_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=DataTransforamtion_config)
                df = self.Col_Transformation(df = self.dataset, data_transformation=data_transformation)
                df, category_mappings = data_transformation.convert_categorical_to_numerical(df = df)
                selected_list = data_transformation.Recuresive_Feature_Elimination(df = df, no_features_to_select=10)

                data_transformation.train_test_splitting(data= df[selected_list])
                logger.info(f"All transformation are completed here.")
                
        except Exception as e:
            logger.exception(e)
            raise e
        

if __name__ == "__main__":

    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<<")
        obj = DataTransformationTrainingConfig()
        val= obj.main()
        print(val)
        logger.info(f">>>>>>>> stage {STAGE_NAME} completed. <<<<<<\n\nx===========x")
    
    except Exception as e:
        logger.exception(e)
        raise e