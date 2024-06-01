import os
from box.exceptions import BoxValueError
import yaml
from mlProject import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any


@ensure_annotations
def read_yaml(path: Path) -> ConfigBox:
    """
    Reading the YAML file

    Args:
        path (str) : file path as input

    Raises:
        e: empty file
        ValueError: if YAML file is empty

    Returns:
        ConfigBox : ConfigBox type
    """

#     try:
#         with open(path) as p:
#             content = yaml.safe_load(p)
#             logger.info(f"yaml file: {path} loaded successfully.")
#             return ConfigBox(content)
#     except BoxValueError:
#         raise ValueError("yaml file is empty.")
#     except Exception as e:
#         raise e

# def read_yaml(path: Path):
    try:
        with path.open("r") as file:  # Open the file correctly
            content = yaml.safe_load(file)
            logger.info(f"yaml file: {path} loaded successfully.")
            return ConfigBox(content)
    except yaml.YAMLError as e:
        logger.error(f"Error while loading yaml file: {path}. Error: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error: {e}")
        raise e

    
@ensure_annotations
def create_directories(filepath: list, verbose = True):
    """
    Create list of dict

    Args:
        filepath (list) : list of path
        ignore_log (bool, optional) : ignore if multiple dirs is to created Default is False

    """

    for path in filepath:
        os.makedirs(path, exist_ok= True)
        if verbose:
            logger.info(f"created directory to : {path}")


@ensure_annotations
def save_json(path: Path, data:dict):
    """
    save the json

    Args: 
        path (path) : path to json file
        data (dict) : data to be saved in json file
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    
    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load the json

    Args:
        path (path) : path to json file 
    
    Returns:
        ConfigBox : data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(path)
    
    logger.info(f"json file Successfully loaded in {path}")

    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path : Path):
    """save binary file

    Args:
        data (any) : data to be saved as binary
        path (Path) : path to saved in the directory
    
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) ->Any:
    """Load binary data

    Args:
        path (path) : path to library file

    Returns:
        Any: Object Stored in the file
    
    """
    data = joblib.load(path)
    logger.info(f'binary file loaded from : {path}')
    return data


@ensure_annotations
def get_size(path: Path):
    """Get size of the file

    Args:
        path (Path) : path to the file

    Returns:
        str: size in KB
    
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f'~{size_in_kb} KB'