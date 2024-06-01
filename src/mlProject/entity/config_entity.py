from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestingConfig:
    root_dir : Path
    source_url : str
    local_data_file : Path
    unzip_dir : Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir:Path
    STATUS_FILE: str
    unzip_data_dir :Path
    all_schema: dict
    target_column : dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    all_schema: dict
    target_column : dict

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path : Path
    model_name : str
    target_columns: str
    model_dir : Path
    


