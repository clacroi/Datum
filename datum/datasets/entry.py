"""
Generic module defining the 2 core components of a Datum dataset : Entry and Observable.
See dataset.py documentation for more details about these classes.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from datum.utils.exceptions import MissingAttributes


class Entry():
    """Class to represent a generic entry in a dataset.
    Attributes:
        _data : dict for representing entries data (attribute, value) pairs
    """

    def __init__(self, data: Dict[str, Any],
                 idx: Optional[int] = None,
                 name: Optional[str] = None,
                 obs_ids: Optional[List[int]] = None) -> None:
        self._data: Dict = data
        if idx is not None:
            self._data['idx'] = idx
        if name is not None:
            self._data['name'] = name
        if obs_ids is not None:
            self._data['obs_ids'] = obs_ids

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        d = {'idx': self.idx,
             'name': self.name,
             'obs_ids': self.obs_ids}
        d['data'] = {}
        for attr_name, attr_value in self._data.items():
            if isinstance(attr_value, Path):
                d['data'][attr_name] = str(attr_value)
            else:
                d['data'][attr_name] = attr_value
        return d

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]):
        """Load parameters from dictionary."""
        return cls(**dictionary)

    @property
    def idx(self) -> Optional[int]:
        """Entry idx."""
        idx = self._data.get('idx', None)
        return idx

    @property
    def name(self) -> Optional[str]:
        """Entry name."""
        name = self._data.get('name', None)
        return name

    @property
    def obs_ids(self) -> List[int]:
        """Entry list of observables ids it is linked to."""
        obs_ids = self._data.get('obs_ids', [])
        return obs_ids

    @property
    def attributes(self):
        """Entry list of attributes."""
        return list(self._data.keys())

    def _add_obs(self, obs_id: int) -> None:
        """Add an obs id into entry list of linked observable ids (obs_ids attribute).
        Args:
            obs_id: Id of observable to add to entry obs_ids.
        """
        if 'obs_ids' in self._data:
            self._data['obs_ids'].append(obs_id)
        else:
            self._data['obs_ids'] = [obs_id]

    def _remove_obs(self, obs_id: int) -> None:
        """Remove an obs id from entry list of linked observable ids (obs_ids attribute).
        Args:
            obs_id: Id of observable to add to entry obs_ids.
        """
        self._data['obs_ids'].remove(obs_id)

    def _update_data(self, data: Dict[str, Any]) -> None:
        """Update entry (attribute, value) pairs in _data with data.
        Args:
            data: dict to update entry data with.
        """
        self._data.update(data)

    def check_attributes(self, attributes: List[str], verbose: bool = True):
        """Checks if all attributes from input list are in entry data.
        Args:
            attributes: list of attributes/keys to check.
            verbose: whether to print missing attributes or not.
        """
        valid = True
        for attr in attributes:
            if attr not in self._data and verbose:
                print('Attribute "{}" not in entry {}"'.format(attr, self))
                valid = False
        return valid

    def __getitem__(self, attribute: str) -> Any:
        """Gets entry value for input attribute if exists."""
        if attribute in self._data:
            return self._data[attribute]
        else:
            raise MissingAttributes(self, [attribute])

    def __contains__(self, attribute: str) -> bool:
        return True if attribute in self._data else False

    def __repr__(self) -> str:
        """Represents an entry through its data (attribute, value) pairs (dictionary)."""

        return str(self._data)


class Observable(Entry):
    """Class to represent an Observable in a dataset.
    """
    def __init__(self, data: Dict[str, Any],
                 idx: Optional[int] = None,
                 entry_id: Optional[int] = None,
                 typ: Optional[str] = None) -> None:
        super().__init__(data, idx=idx)
        if entry_id is not None:
            self._data['entry_id'] = entry_id
        if typ:
            self._data['type'] = typ
        else:
            if 'type' not in data:
                raise Exception('Observable type not passed through init argument typ'
                                ' neither data ("type", value) entry')

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        d = {'idx': self.idx,
             'entry_id': self.entry_id,
             'typ': self.typ}
        d['data'] = {}
        for attr_name, attr_value in self._data.items():
            if isinstance(attr_value, Path):
                d['data'][attr_name] = str(attr_value)
            else:
                d['data'][attr_name] = attr_value
        return d

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]):
        """Load parameters from dictionary."""
        return cls(**dictionary)

    @property
    def entry_id(self) -> Optional[int]:
        """Observable related entry id."""
        entry_id = self._data.get('entry_id', None)
        return entry_id

    @property
    def typ(self) -> Optional[str]:
        """Observable type."""
        typ = self._data.get('type', None)
        return typ
