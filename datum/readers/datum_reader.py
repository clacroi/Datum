"""
Generic module to populate a Dataset object from a Datum-formatted dataset.
"""

from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import json
import pickle

from datum.datasets import Dataset, Entry, Observable
from datum.readers import DatasetReader


def load_datum_dataset(root_dir: Path) -> Dataset:
    with open(root_dir / 'dataset.json', 'r') as out_file:
        dataset_json_dict = json.load(out_file)
    dataset = Dataset.from_dict(dataset_json_dict)

    reader = DatumReader(root_dir)
    reader.feed(dataset)
    return dataset


class DatumReader(DatasetReader):
    """Class to populate a Dataset object from a VOC-like detection database.

    Attributes:
    """

    def __init__(self, root_dir: Path) -> None:
        super().__init__(root_dir)
    
    def feed(self, dataset: Dataset, clear_existing_data: bool=False,
             entry_format: Optional[Dict[str, Any]] = None,
             obs_format: Optional[Dict[str, Any]] = None) -> None:
        if clear_existing_data:
            dataset.clear_data()
        entry_dataset_ids = {}

        # load entry format if exists
        if (self.root_dir / '.pickled' / 'entry_format.pickle').exists():
            with open((self.root_dir / '.pickled' / 'entry_format.pickle'), 'rb') as handle:
                entry_format = pickle.load(handle)
            print('Datum reader will type entries attributes as follow :\n{}'.format(entry_format))
        else:
            entry_format = None

        # load obs format if exists
        if (self.root_dir / '.pickled' / 'obs_format.pickle').exists():
            with open((self.root_dir / '.pickled' / 'obs_format.pickle'), 'rb') as handle:
                obs_format = pickle.load(handle)
            print('Datum reader will type observables attributes as follow :\n{}'.format(obs_format))
        else:
            obs_format = None

        with open(self.root_dir / 'entries.json', 'r') as out_file:
            entries_json = json.load(out_file)
        for entry_json in entries_json:
            # Format attributes values loaded from json
            if entry_format:
                for attr, val in entry_json['data'].items():
                    if attr in entry_format:
                        entry_json['data'][attr] = entry_format[attr](val)

            entry = Entry.from_dict(entry_json)
            entry_dataset_ids[entry.idx] = dataset.add_entry(entry)

        with open(self.root_dir / 'observables.json', 'r') as out_file:
            observables_json = json.load(out_file)
        for obs_json in observables_json:
            if obs_format:
                for attr, val in obs_json['data'].items():
                    if attr in obs_format:
                        entry_json['data'][attr] = obs_format[attr](val)
    
            obs = Observable.from_dict(obs_json)
            dataset.add_observable(entry_dataset_ids[obs.entry_id], obs)
