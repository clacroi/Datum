"""
Generic module to represent a dataset Formatter.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

from datum.datasets import Dataset


class DatasetFormatter(ABC):
    """Abstract class to represent a Dataset formatter for writing/saving on disk a Dataset object
    at a specific format.

    Attributes:
        root_dir: Path of root directory to save dataset
    """
    def __init__(self, root_dir: Path):
        self.root_dir: Path = root_dir
        super().__init__()

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        return {'formatter_type': type(self).__name__,
                'root_dir': self.root_dir}

    @abstractmethod
    def format(self, dataset: Dataset) -> None:
        pass
