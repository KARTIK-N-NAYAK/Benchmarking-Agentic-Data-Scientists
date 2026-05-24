import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class DataExtension(Enum):
    CSV = ".csv"
    PKL = ".pkl"
    TXT = ".txt"
    NPY = ".npy"


@dataclass
class DataDescription:
#    features: List
#    num_features: int
    target_cols: List
#    description: str
    feature_types: Dict[str, str]
#    target_types: Dict[str, str]
#    feature_values: Optional[Dict] = None


@dataclass
class MetaData:
    title: str
    kaggle_name: str
#    aux_data_names: List[str]
#    aux_data_formats: List[DataExtension]
#    raw_description: str
#    task_description: str
#    task_type: str
    data_description: DataDescription
    prediction_type: str
    input_types: List[str]
#    metric_path: Optional[str]
    score_name: str
#    metric_description: str
#    positive_class_name: str
    id_col: str
#    train_data_name: str = "train_X"
#    train_data_format: DataExtension = DataExtension.CSV
#    test_data_name: str = "test_X"
#    test_data_format: DataExtension = DataExtension.CSV
#    train_target_name: Optional[str] = "train_y"
#    train_target_format: Optional[DataExtension] = DataExtension.CSV
#    test_target_name: Optional[str] = "test_y"
#    test_target_format: Optional[DataExtension] = DataExtension.CSV
#    lgbm_objective: Optional[str] = None

    def __post_init__(self):
        """Used to force any format that is not an instance of DataExtension, into it"""
#        if not isinstance(self.train_data_format, DataExtension):
#            self.train_data_format = DataExtension(self.train_data_format)
#        if not isinstance(self.train_target_format, DataExtension):
#            self.train_target_format = DataExtension(self.train_target_format)
#        if not isinstance(self.test_data_format, DataExtension):
#            self.test_data_format = DataExtension(self.test_data_format)
#        if not isinstance(self.test_target_format, DataExtension):
#            self.test_target_format = DataExtension(self.test_target_format)

    def save(self, save_path: str | Path):
        """Save the metadata as a json

        Args:
            save_path (str | Path): save path
        """
        save_path = Path(save_path)
        metadata_dict = asdict(self)
        metadata_dict["data_description"] = asdict(metadata_dict["data_description"])
        for key in metadata_dict:
            if isinstance(metadata_dict[key], DataExtension):
                metadata_dict[key] = metadata_dict[key].value

        json.dump(
            obj=metadata_dict,
            indent=2,
            fp=open(
                save_path / "metadata.json",
                "w",
            ),
        )

    def asdict(self) -> dict:
        """Returns the metadata as a dictionary

        Returns:
            dict: _description_
        """
        metadata_dict = asdict(self)
        metadata_dict["data_description"] = asdict(metadata_dict["data_description"])
        return metadata_dict


def load_metadata_from_json(load_path: str | Path, as_dict: bool = False) -> MetaData | Dict:
    """Loads metadata from json

    Args:
        load_path (str | Path): Load path
        as_dict (bool): if true the metadata is returned as a dictionary. Default: False

    Returns:
        MetaData: loaded metadata
    """
    load_path = Path(load_path) / "metadata.json"
    metadata_dict = json.load(open(load_path))
    metadata_dict["data_description"] = DataDescription(**metadata_dict["data_description"])
    metadata = MetaData(**metadata_dict)
    if as_dict:
        return metadata.asdict()
    return metadata


def make_metadata_injectable(metadata: dict) -> dict:
    """Function to change lists into strings in the metadata. Useful for injecting metadata lists into
    code templates"""
    for key in metadata:
        if isinstance(metadata[key], list):
            metadata[key] = ", ".join(map(str, metadata[key]))
        elif isinstance(metadata[key], dict):
            metadata[key] = make_metadata_injectable(metadata[key])
        elif is_dataclass(metadata[key]):
            metadata[key] = make_metadata_injectable(asdict(metadata[key]))
    return metadata
