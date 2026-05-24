import json
from pathlib import Path


class MetaData:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        metadata_path = Path(dataset_path) / 'data' / 'metadata.json'

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self._prediction_type = metadata['prediction_type']
        self._target_cols = metadata['data_description']['target_cols']
        self._id_field = metadata['id_col']

    @property
    def prediction_type(self):
        return self._prediction_type
    @property
    def target_cols(self):
        return self._target_cols
    @property
    def id_field(self):
        return self._id_field
