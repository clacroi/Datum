"""
Generic module to format and save a Dataset under Datum format.
Datum format is defined by a set of minimal construction rules :
- 1 directory per dataset, containing
-- 1 JSON file dataset.json containing dataset serialized attributes (except entries/observables)
-- 1 JSON file entries.json containing serialized entries
-- 1 JSON file observables.json containing serialized observables 
"""

from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import json
import pickle
import os

from datum.datasets import Dataset, Entry
from datum.formatters import DatasetFormatter


class DatumFormatter(DatasetFormatter):
    """Class to save a Dataset object under Datum dataset format.

    Attributes:
        _images_dir: path to VOC images folder.
        _annotations_dir: path to VOC annotations files folder.
        _images_annotations: list of images annotations.
        _obs_annotations: list of observables annotations.
    """
    def __init__(self, root_dir: Path) -> None:
        super().__init__(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        return {'formatter_type': type(self).__name__,
                'root_dir': str(self.root_dir)}

    def format(self, dataset: Dataset,
               entry_format: Optional[Dict[str, Any]] = None,
               obs_format: Optional[Dict[str, Any]] = None) -> None:
        """Saves input dataset under VOC-Detection format.
        Args:
            dataset: dataset object format and save.
            copy_images: whether to save or not images on disk.
        """
        dataset_json_dict = dataset.to_dict()
        with open(self.root_dir / 'dataset.json', 'w') as out_file:
            json.dump(dataset_json_dict, out_file, indent=4)

        with open(self.root_dir / 'entries.json', 'w') as out_file:
            json.dump([entry.to_dict() for entry_id, entry in  dataset._entries.items()], out_file, indent=1)

        with open(self.root_dir / 'observables.json', 'w') as out_file:
            json.dump([obs.to_dict() for obs_id, obs in  dataset._observables.items()], out_file, indent=1)

        if entry_format or obs_format:
            (self.root_dir / '.pickled').mkdir(exist_ok=True)
        if entry_format:
            with open((self.root_dir / '.pickled' / 'entry_format.pickle'), 'wb') as handle:
                pickle.dump(entry_format, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if obs_format:
            with open((self.root_dir / '.pickled' / 'entry_format.pickle'), 'wb') as handle:
                pickle.dump(entry_format, handle, protocol=pickle.HIGHEST_PROTOCOL)
