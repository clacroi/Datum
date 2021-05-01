"""
Abstract class to represent a database reader used to populate a Dataset object.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

# TODO : typing causes cyclic import and import error, find solution for fixing Dataset import.
#from datum.datasets import Dataset
#from datum.datasets import Dataset


class StaticTransformer(ABC):
    """Abstract class used for representing a dataset (Dataset) static transformer, i.e an object
    used to modify all dataset entries (and corresponding observables) sequentially
    and in one shot.
    """
    FORBIDDEN_ENTRIES_ATTRIBUTES = set(['name'])
    FORBIDDEN_OBS_ATTRIBUTES = set(['type', 'entry_id'])
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, dataset) -> None:
        pass
