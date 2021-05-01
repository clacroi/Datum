"""
Abstract class to represent a database reader used to populate a Dataset object.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from datum.utils.annotations_utils import get_dict_value
from datum.datasets import Dataset


class AttributeConstructor():
    """Simple class for representing a dataset attribute constructor : name of constructed
    attribute, construction process, attribute value type, attribute default value, if its
    value should be set to its default value if the attribute value can't be extracted,
    its lookup path in a hierarchical dict structure. This class does not represent the value
    of the attribute itself.

    Attributes:
        _name: constructed attribute name
        _process ("lookup" or "custom"): the attribute construction process type.
        _typ (Python type): constructed attribute value type.
        _default_val (_typ): constructed attribute default value.
        _set_to_default: whether constructed attribute value should be set to
            default if attribute.
        can't be extracted with its construction process.
        _lookup_path: path in input tree-like structure to get to constructed attribute value.
    """
    def  __init__(self, name: str, process: str,
                  typ: type, default_val: Any, set_to_default: bool,
                  lookup_path : Optional[List[str]] = None) -> None:
        self._name: str = name
        if process not in ['lookup', 'custom']:
            raise ValueError
        if process == 'lookup' and not lookup_path:
            raise ValueError
        self._process: str  = process
        self._typ: type = typ
        self._default_val: Any = default_val
        self._set_to_default: bool = set_to_default
        if lookup_path:
            self._lookup_path: List[str] = lookup_path
        else:
            self._lookup_path: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        return {'attribute_constructor_type': type(self).__name__,
                'name': self._name,
                'process': self._process,
                'typ': self._typ,
                'default_val': self._default_val,
                'set_to_default': self._set_to_default,
                'lookup_path': self._lookup_path}

    @property
    def name(self):
        return self._name
    @property
    def process(self):
        return self._process
    @property
    def typ(self):
        return self._typ
    @property
    def set_to_default(self):
        return self._set_to_default
    @property
    def default_val(self):
        return self._default_val
    @property
    def lookup_path(self):
        return self._lookup_path

    def extract_in(self, data: Dict[str, Any],
                   lookup_path: Optional[List[str]] = None) -> Tuple[bool, Optional[type]]:
        """Extracts value of attribute in input tree-like structure
        represented by a dictionary (dictionary of dictionary or list of dictionaries).

        Args:
            data: input tree-like structure to look attribute value for.
            lookup_path: optional argument overriding constructor own lookup_path.
        Returns:
            bool indicating whether extraction executed correctly.
            extracted value for the attribute.
        """
        if self.process != 'lookup':
            raise Exception('Attribute has no "lookup" construction process type')
        if lookup_path:
            val = get_dict_value(data, lookup_path)
        else:
            val = get_dict_value(data, self.lookup_path)
        if val is None:
            if not self.set_to_default:
                return False, None
            else:
                return True, self.default_val
        try:
            val = self.typ(val)
        except ValueError:
            return False, val
        return True, val

    def __str__(self):
        return '(name="{}", process={}, type={}, default_val={}, set_to_default={}, '\
               'lookup_path={})'.format(self.name, self.process, self.typ,
                                        self.default_val, self.set_to_default, self.lookup_path)


def update_entries_attr_constructors(default_constructors: List[AttributeConstructor],
                                     constructors: List[AttributeConstructor]) \
                                     -> List[AttributeConstructor]:
    """Update/override default list of attributes constructors with user constructors.

    Args:
        default_constructors: default list of constructors to update.
        constructors: list of user input constructors.

    Returns:
        updated list of constructors.
    """
    updated_constructors = constructors
    for default_attr in default_constructors:
        to_add = True
        for attr in constructors:
            if default_attr.name == attr.name:
                to_add = False
                break
        if to_add:
            updated_constructors.append(default_attr)
    return updated_constructors


class ObsAttributeConstructor(AttributeConstructor):
    """Simple class for representing a dataset observable attribute constructor :
    type of observable it is related to, constructed attribute name, construction process,
    value type, default value, if its value should be set to its default value if the attribute
    value can't be extracted, its lookup path in a hierarchical dict structure.

    Attributes:
        _observable_type: observable type the attribute is related to
    """
    def __init__(self, observable_type: type, name: str, process: str,
                 typ: type, default_val: Any, set_to_default: bool,
                 lookup_path: Optional[List[str]] = None) -> None:
        super().__init__(name, process, typ, default_val, set_to_default, lookup_path)
        self._observable_type: str = observable_type

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        return {'attribute_constructor_type': type(self).__name__,
                'observable_type': self._observable_type,
                'name': self._name,
                'process': self._process,
                'typ': self._typ,
                'default_val': self._default_val,
                'set_to_default': self._set_to_default,
                'lookup_path': self._lookup_path}

    @property
    def observable_type(self):
        return self._observable_type


def update_obs_attr_constructors(default_constructors: List[AttributeConstructor],
                                 constructors: List[AttributeConstructor]) \
                                 -> List[AttributeConstructor]:
    """Update/override default list of observables attributes constructors with user constructors.

    Args:
        default_constructors: default list of constructors to update.
        constructors: list of user input constructors.

    Returns:
        updated list of constructors.
    """
    updated_constructors = constructors
    for default_attr in default_constructors:
        to_add = True
        for attr in constructors:
            if (default_attr.observable_type == attr.observable_type) \
                    and (default_attr.name == attr.name):
                to_add = False
                break
        if to_add:
            updated_constructors.append(default_attr)
    return updated_constructors


class DatasetReader(ABC):
    """Abstract class to represent a Dataset reader for populating
    a Dataset object from data and annotation files on a disk.

    Attributes:
        root_dir: Path of root directory to read database from
    """
    def __init__(self, root_dir: Path,
                 entries_constructors: List[AttributeConstructor]=[],
                 obs_constructors: List[AttributeConstructor]=[]) -> None:
        self.root_dir: Path = root_dir
        self._entries_constructors: List = entries_constructors
        self._obs_constructors: List = obs_constructors
        super().__init__()

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        d = {'reader_type': type(self).__name__,
             'root_dir': self.root_dir}
        if self._entries_constructors:
            d['entries_constructors'] = [const.to_dict() for const in self._entries_constructors]
        if self._obs_constructors:
            d['obs_constructors'] = [const.to_dict() for const in self._obs_constructors]
        return d

    @abstractmethod
    def feed(self, dataset: Dataset, clear_existing_data: bool=False) -> None:
        pass
