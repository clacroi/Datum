"""
Generic module to represent, construct and iterate over a dataset.

A generic Entry is a set of (key, value) pairs or (attribute, value) pairs represented by a
Python dictionary. It has two special keys : idx, name and obs_ids that are always defined when
an entry is added to a Dataset and can be directly accessed through class properties.

An observable is a special kind of Entry with 2 additional special keys : an entry_id and a type.

A dataset is defined by two sets of entries :
- The first set contains elements of type Entry that are the "primary" entries of the dataset.
- The second set contains elements of type Observable, also called "secondary" entries.

A Dataset should be populated by a client using functions add_entry(), add_observable().
During construction, the dataset makes sure that each added observable is linked to a
primary entry through its "entry_id" attribute. It also makes sure that in this case, its
corresponding primary entry with id entry_id is linked to base observable through attribute obs_id
(a list containing all the Observable objects ids the primary entry is linked with).

Clients/Readers must populate a dataset with entries and observables having the minimal
required set of attributes :

- An entry can be added to a dataset iff it has the dataset required entries attributes in
  _min_entries_attributes

- A observable can be added to a dataset iff
-- it has dataset required observables common attributes (in _min_obs_common_attributes)
-- it has dataset required per-type observable attributes (in _min_obs_type_attributes)

Entries attributes can be modified with function update_entry_data() while its id and name
can't be modified.

Observable attributes can be modified with function update_observable_data() while it id
and 'entry_id' can't be modified

Entries and observables can be removed with functions remove_entry() and remove_attribute()

Use example : see VocDetectionReader class for an example of how to populate a dataset and its
'__main__' section for example of how to use the Dataset object.
"""

from typing import Union, Tuple, List, Set, Iterable, Dict, Any, Optional

import pandas as pd

from datum.datasets import Entry, Observable
from datum.transformers import EntryMapper, ObservableMapper
from datum.utils.exceptions import MissingAttributes, MissingEntry, ForbiddenAttribute, EntryAlreadyExists


class Dataset:
    """Class to represent a dataset and iterate over it.

    Attributes:
        _next_entry_id: next available entry id.
        _next_observable_id: next available observable id.
        _entries: Dictionary mapping entries ids to entries names, data (Dictionary)
            and related observables ids (List of int).
        _observables: Dictionary mapping observables ids.
            to observables data (Dictionary)
        _names_ids_map: Dictionary mapping entries names to entries ids.
        _min_entries_attributes: list of dataset required entries attributes
        _min_obs_common_attributes: list of dataset required (common) observables attributes
        _min_obs_type_attributes: dict mapping observables types (str) to per-type
            observables attributes
        new_entries_mappers: list of entry mappers to apply at each new entry adding.
        new_obs_mappers: list of observable mappers to apply at each new observable adding.
    """

    # Default dataset entries and observables common/per-type attributes
    # (empty for generic dataset)
    DEFAULT_MIN_ENTRIES_ATTRIBUTES = []
    DEFAULT_MIN_OBS_COMMON_ATTRIBUTES = []
    DEFAULT_MIN_OBS_TYPE_ATTRIBUTES = {}

    def __init__(self, name: Optional[str] = 'Unnamed',
                 min_entries_attributes: Optional[List[str]] = None,
                 min_obs_common_attributes: Optional[List[str]] = None,
                 min_obs_type_attributes: Optional[Dict[str, List[str]]] = None,
                 new_entry_mappers: Optional[List[EntryMapper]] = None,
                 new_obs_mappers: Optional[List[ObservableMapper]] = None):
        self.name = name
        self._next_entry_id: int = 0
        self._next_observable_id: int = 0
        self._entries: Dict[int, Entry] = {}
        self._observables: Dict[int, Observable] = {}
        self._names_ids_map: Dict[str, int] = {}
        if min_entries_attributes:
            self._min_entries_attributes: List = min_entries_attributes
        else:
            self._min_entries_attributes: List[str] = []
        if min_obs_common_attributes:
            self._min_obs_common_attributes: List = min_obs_common_attributes
        else:
            self._min_obs_common_attributes: List[str] = []
        if min_obs_type_attributes:
            self._min_obs_type_attributes: Dict[str, List[str]] = min_obs_type_attributes
        else:
            self._min_obs_type_attributes: Dict[str, List[str]] = {}

        if new_entry_mappers:
            self.new_entry_mappers: List[EntryMapper] = new_entry_mappers
        else:
            self.new_entry_mappers: List[EntryMapper] = []

        if new_obs_mappers:
            self.new_obs_mappers: List[ObservableMapper] = new_obs_mappers
        else:
            self.new_obs_mappers: List[ObservableMapper] = []

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        # TODO : add mappers
        return {'name': self.name,
                'min_entries_attributes': self.min_entries_attributes,
                'min_obs_common_attributes': self.min_obs_common_attributes,
                'min_obs_type_attributes': self.min_obs_type_attributes}

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]):
        """Load parameters from dictionary."""
        return cls(**dictionary)

    @property
    def entries_df(self) -> pd.DataFrame:
        """Returns a pandas DataFrame with rows corresponding to dataset entries and columns
        to all entries existing attributes + entries names.
        """
        entries = {entry_id: entry._data for (entry_id, entry) in self._entries.items()}
        df = pd.DataFrame.from_dict(entries, orient='index')
        return df

    def construct_entries_df(self, target_attributes: List[str] = None):
        """Returns a pandas DataFrame with rows corresponding to dataset entries and columns
        to entries existing attributes in target_attributes.
        """
        if target_attributes:
            entries = {}
            for (entry_id, entry) in self._entries.items():
                entries[entry_id] = {key: val for key, val in entry._data.items() if key in target_attributes}
            return pd.DataFrame.from_dict(entries, orient='index')
        else:
            return self.entries_df

    @property
    def observables_df(self) -> pd.DataFrame:
        """Returns a pandas DataFrame with rows corresponding to dataset observables and columns
        to all observables existing attributes + entries names.
        """
        observables = {obs_id: obs._data for (obs_id, obs) in self._observables.items()}
        return pd.DataFrame.from_dict(observables, orient='index')

    @property
    def entries_idxs_list(self) -> List[int]:
        """List of entries names."""
        return list(self._entries.keys())

    @property
    def entries_names_list(self) -> List[str]:
        """List of entries names."""
        return list(self._names_ids_map.keys())

    @property
    def entries_names_set(self) -> Set[str]:
        """Set of entries names."""
        return set(self._names_ids_map.keys())

    @property
    def min_entries_attributes(self) -> List:
        """Minimal entries attributes list."""
        return self._min_entries_attributes

    @property
    def min_obs_common_attributes(self) -> List:
        """Minimal observables common attributes list."""
        return self._min_obs_common_attributes

    @property
    def min_obs_type_attributes(self) -> dict:
        """Minimal observables per-type attributes list."""
        return self._min_obs_type_attributes

    def infer_entry_id_name(self, entry_token: Union[int, str]) -> Tuple[int, str]:
        """Returns (id, name) of entry corresponding to input token (either entry id or name)

        Args:
            entry_token: entry id or name
        Returns
            entry (id, name) corresponding to entry with id/name = entry_token
        """
        if isinstance(entry_token, int):
            if entry_token not in self._entries:
                raise MissingEntry(entry_token, 'Entry')
            entry_id = entry_token
            entry_name = self._entries[entry_id].name
        elif isinstance(entry_token, str):
            if entry_token not in self._names_ids_map:
                raise MissingEntry(entry_token, 'Entry')
            entry_id = self._names_ids_map[entry_token]
            entry_name = entry_token
        else:
            raise TypeError
        return entry_id, entry_name

    def add_entry(self, entry: Union[Entry, Dict[str, Any]],
                  observables: Optional[List[Union[Observable, Dict[str, Any]]]] = None) -> int:
        """Adds an entry to dataset and observables related to it
        Args:
            entry: entry to add to dataset.
            observables: List of observables (Observable/Dict)
                to associate with entry.
        Returns:
            Id given to entry in dataset.
        """
        if not isinstance(entry, Entry):
            if 'name' not in entry:
                entry = Entry(entry, idx=self._next_entry_id, name=str(self._next_entry_id))
            else:
                entry = Entry(entry, idx=self._next_entry_id)
        else:
            # Forget about entry id and obs_ds as they are managed by dataset
            update = {'idx': self._next_entry_id, 'obs_ids': []}
            if entry.name is None:
                update['name'] = str(self._next_entry_id)
            entry._update_data(update)

        if not entry.check_attributes(self._min_entries_attributes):
            raise MissingAttributes(entry)
        if entry.name in self._names_ids_map:
            raise EntryAlreadyExists(entry.name, 'Entry')

        self._entries[self._next_entry_id] = entry
        self._names_ids_map[entry.name] = self._next_entry_id

        # Apply mappers from new_entry_mappers
        for mapper in self.new_entry_mappers:
            mapper_out = mapper.apply_to(entry, apply_mode='force')
            if mapper_out:
                self.update_entry_data(entry.idx, mapper_out)

        if observables:
            for observable in observables:
                self.add_observable(entry.idx, observable)
        self._next_entry_id += 1

        return self._next_entry_id - 1

    def remove_entry(self, entry_token: Union[int, str]) -> None:
        """Removes an entry from dataset
        Args:
            entry_token: Id or name of entry to remove.
        """
        entry_id, entry_name = self.infer_entry_id_name(entry_token)
        # Remove associated observables
        [self._observables.pop(obs) for obs in self._entries[entry_id].obs_ids]
        # Remove entry name
        self._names_ids_map.pop(entry_name)
        # Remove entry
        self._entries.pop(entry_id)

    def update_entry_data(self, entry_token: Union[int, str], entry_data: Dict[str, Any]) -> None:
        """Updates an entry attributes in dataset
        Args:
            entry_token: Id/name of entry to update in dataset.
            entry_data: Attributes of entry to update.
        """
        entry_id, entry_name = self.infer_entry_id_name(entry_token)
        if entry_data.get('idx', entry_id) != entry_id:
            raise ForbiddenAttribute(self._entries[entry_id], 'idx')

        self._entries[entry_id]._update_data(entry_data)
        # If entry name is modified, modify name to id map accordingly
        if entry_data.get('name', entry_name) != entry_name:
            self._names_ids_map[entry_data['name']] = entry_id
            self._names_ids_map.pop(entry_name)

    def add_observable(self, entry_token: Union[int, str],
                       observable: Union[Observable, Dict[str, Any]]) -> int:
        """Adds observable to dataset
        Args:
            entry_token: Id/name of entry related to observable to add to dataset.
            observable: observable (Observable or dict of attributes)
                to add to dataset.
        Returns:
            Id given to observable in dataset.
        """
        entry_id, _ = self.infer_entry_id_name(entry_token)

        if not isinstance(observable, Observable):
            observable = Observable(observable, idx=self._next_observable_id, entry_id=entry_id)
        else:
            # Forget about observable id and entry_id as they are managed by dataset
            observable._update_data({'idx': self._next_observable_id, 'entry_id': entry_id})

        # Check that observable has common/type-specific obs attributes
        type_attributes = self._min_obs_type_attributes.get(observable.typ, [])
        if not (observable.check_attributes(self._min_obs_common_attributes) \
            and observable.check_attributes(type_attributes)) :
            raise MissingAttributes(observable)
        self._observables[self._next_observable_id] = observable
        self._entries[entry_id]._add_obs(self._next_observable_id)

        # Apply mappers from new_obs_mappers
        for mapper in self.new_obs_mappers:
            if mapper.obs_type != 'all' and mapper.obs_type != observable.typ:
                continue
            mapper_out = mapper.apply_to(observable,
                                         entry=self._entries[entry_id],
                                         apply_mode='force')
            if mapper_out:
                self.update_observable_data(observable.idx, mapper_out)

        self._next_observable_id += 1

        return self._next_observable_id - 1

    def remove_observable(self, obs_id: int) -> None:
        """Removes an observable from dataset.
        Args:
            obs_id: Id of observable to remove.
        """
        if obs_id not in self._observables:
            raise MissingEntry(obs_id, 'Observable')
        entry_id = self._observables[obs_id].entry_id

        # Remove observable id in corresponding entry
        self._entries[entry_id]._remove_obs(obs_id)
        # Remove observable
        self._observables.pop(obs_id)

    def update_observable_data(self, obs_id: int, obs_data: Dict[str, Any]) -> None:
        """Updates an observable attributes in dataset
        Args:
            obs_id: Id of observable to update in dataset.
            obs_data: Attributes of observable to update.
        """
        if obs_id not in self._observables:
            raise MissingEntry(obs_id, 'Observable')
        if obs_data.get('idx', self._observables[obs_id].idx) != self._observables[obs_id].idx:
            raise ForbiddenAttribute(self._observables[obs_id], 'idx')
        if obs_data.get('entry_id', self._observables[obs_id].entry_id) \
                != self._observables[obs_id].entry_id:
            raise ForbiddenAttribute(self._observables[obs_id], 'entry_id')
        if obs_data.get('type', self._observables[obs_id].typ) != self._observables[obs_id].typ:
            raise ForbiddenAttribute(self._observables[obs_id], 'type')

        self._observables[obs_id]._update_data(obs_data)

    def remove_entries_without_obs(self) -> None:
        """Remove entries from dataset without any observable."""
        # Use a copy of self._entries to avoid error :
        # "RuntimeError : dictionary changed size during iteration"
        for _, entry in self._entries.copy().items():
            if len(entry.obs_ids) == 0:
                self.remove_entry(entry.idx)

    def clear_data(self) -> None:
        """Removes existing entries and observables from dataset.
        """
        self._next_entry_id = 0
        self._next_observable_id = 0
        self._entries = {}
        self._observables = {}
        self._names_ids_map = {}

    def __getitem__(self, entry_token: Union[int, str]) -> Tuple[Entry, List[Observable]]:
        """Gets the entry with id/name entry_token and its associated observables."""
        entry_id, _ = self.infer_entry_id_name(entry_token)
        observables = [self._observables[obs_id] for obs_id in self._entries[entry_id].obs_ids]
        return self._entries[entry_id], observables

    def __contains__(self, entry_token: Union[int, str]) -> bool:
        if isinstance(entry_token, int):
            return entry_token in self._entries
        elif isinstance(entry_token, str):
            return entry_token in self._names_ids_map
        else:
            raise TypeError

    def __iter__(self) -> Iterable:
        """Iterates over entries and corresponding observables in dataset."""
        for entry_id, entry in self._entries.items():
            yield entry, [self._observables[obs_id] for obs_id in self._entries[entry_id].obs_ids]

    def __len__(self) -> int:
        """The number of entries in the dataset."""
        return len(self._entries)

    def __add__(self, other):
        """Constructs and returns a new dataset that is the concatenation of self with other.
        Args:
            other (Dataset): Dataset to merge with self.
        Returns:
            New Dataset made of concatenation of other with self.
        """
        dataset = Dataset(min_entries_attributes=self.min_entries_attributes,
                          min_obs_common_attributes=self.min_obs_common_attributes,
                          min_obs_type_attributes=self.min_obs_type_attributes)
        for entry, observables in self:
            dataset.add_entry(entry, observables=observables)
        for entry, observables in other:
            dataset.add_entry(entry, observables=observables)
        return dataset

    @staticmethod
    def merge(datasets: List, attributes_merging: str='intersection'):
        """Return a new dataset that is the fusion between datasets in input dataset list
        (both entries and observables are merged). New dataset min_entries_attributes,
        min_obs_common_attributes and min_obs_type_attributes are merged by intersection
        or union.

        Args:
            datasets (list of Datasets): List of datasets to merge together.
            attributes_merging: Attributes merging method (intersection/union).
        Returns:
            Dataset made of merged input datasets.
        """
        if attributes_merging not in['intersection', 'union']:
            raise Exception('attributes_merging should be "intersection" or "union"')
        if len(datasets) == 0:
            return None
        entries_attributes = set(datasets[0].min_entries_attributes)
        obs_common_attributes = set(datasets[0].min_obs_common_attributes)
        obs_type_attributes = datasets[0].min_obs_type_attributes
        obs_type_attributes = {t: set(l) for t, l in obs_type_attributes.items()}

        # Construct merged dataset min_entries_attributes, min_obs_common_attributes
        # and min_obs_type_attributes
        for dataset in datasets:
            if attributes_merging == 'intersection':
                entries_attributes = entries_attributes.intersection(
                    dataset.min_entries_attributes)
                common_obs_attributes = obs_common_attributes.intersection(
                    dataset.min_obs_common_attributes)
            else:
                entries_attributes = entries_attributes.union(dataset.min_entries_attributes)
                common_obs_attributes = obs_common_attributes.union(
                    dataset.min_obs_common_attributes)

            for t, l in dataset.min_obs_type_attributes.items():
                if t in obs_type_attributes:
                    if attributes_merging == 'intersection':
                        obs_type_attributes[t] = obs_type_attributes[t].intersection(l)
                    else:
                        obs_type_attributes[t] = obs_type_attributes[t].union(l)
                else:
                    obs_type_attributes[t] = l

        obs_type_attributes = {t: list(s) for t, s in obs_type_attributes.items()}
        merged_dataset = Dataset(min_entries_attributes=list(entries_attributes),
                                 min_obs_common_attributes=list(common_obs_attributes),
                                 min_obs_type_attributes=obs_type_attributes)
        for dataset in datasets:
            for entry, observables in dataset:
                merged_dataset.add_entry(entry, observables=observables)

        return merged_dataset

    @classmethod
    def from_dataframes(cls, entries_df: pd.DataFrame, obs_df: pd.DataFrame):
        """Construct a new Dataset object from 2 input dataframes : entries_df for entries
        and obs_df for observables. New dataset min_entries_attributes, min_obs_common_attributes
        and min_obs_type_attributes correspond to input dataframes valid (= not any NA value)
        columns (but other columns values are still integrated to dataframe).
        Entries dataframe and observable dataframes should have valid columns for class-required
        attributes (cls.DEFAULT_MIN_ENTRIES_ATTRIBUTES, cls.DEFAULT_MIN_OBS_COMMON_ATTRIBUTES,
        cls.DEFAULT_MIN_OBS_TYPE_ATTRIBUTES).

        Args:
            entries_df: Entries dataframe.
            obs_df: Observables dataframe.
        Returns:
            Dataset constructed from entries_df and obs_df.
        """
        # Check that entries_df has required columns
        entries_non_null = list(entries_df.columns[~entries_df.isnull().any()])
        for attribute in cls.DEFAULT_MIN_ENTRIES_ATTRIBUTES + ['idx', 'name']:
            if attribute not in entries_df.columns:
                raise MissingAttributes(entries_df, [attribute])
            if attribute not in entries_non_null:
                raise MissingAttributes(entries_df, [attribute])
        # 'idx' and 'name' should not be in min_entries_attributes
        entries_non_null.remove('idx')
        entries_non_null.remove('name')

        # Check that observables_df has required columns
        obs_non_null = list(obs_df.columns[~obs_df.isnull().any()])
        for attribute in cls.DEFAULT_MIN_OBS_COMMON_ATTRIBUTES + ['type', 'entry_id']:
            if attribute not in obs_df.columns:
                raise MissingAttributes(obs_df, [attribute])
            if attribute not in obs_non_null:
                raise MissingAttributes(obs_df, [attribute])
        # 'idx', 'entry_id' and 'type' should not be in min_obs_common_attributes
        obs_non_null.remove('idx')
        obs_non_null.remove('entry_id')
        obs_non_null.remove('type')

        # Check that observables_df has required columns
        obs_type_non_null = {}
        for t, t_attributes in cls.DEFAULT_MIN_OBS_TYPE_ATTRIBUTES.items():
            t_obs_non_null = obs_df.columns[~obs_df[obs_df['type'] == t].isnull().any()]
            t_obs_non_null = list(set(t_obs_non_null) - set(obs_non_null))
            for attribute in t_attributes:
                if attribute not in obs_df.columns:
                    raise MissingAttributes(obs_df, [attribute])
                if attribute not in t_obs_non_null:
                    raise MissingAttributes(obs_df, [attribute])
            obs_type_non_null[t] = t_obs_non_null
        # 'idx', entry_id' and 'type' should not be in min_obs_type_attributes
        for t, min_attributes_t in obs_type_non_null.items():
            min_attributes_t.remove('idx')
            min_attributes_t.remove('entry_id')
            min_attributes_t.remove('type')

        # Check input dataframes consistency
        if len(entries_df['name'].unique()) != len(entries_df):
            raise Exception('All entries names should be unique')
        if set(obs_df['entry_id'].unique()) - set(entries_df.index):
            raise Exception('All observable should have valid "entry_id"')
        dataset = Dataset(min_entries_attributes=entries_non_null,
                          min_obs_common_attributes=obs_non_null,
                          min_obs_type_attributes=obs_type_non_null)
        for index, entry in entries_df.iterrows():
            observables = obs_df[obs_df['entry_id'] == index].drop('entry_id', axis=1)
            dataset.add_entry(entry,
                              observables=[obs_data for _, obs_data in observables.iterrows()])
        return dataset
