"""
Abstract class to represent a database reader used to populate a Dataset object.
"""

from typing import List, Dict, Any, Optional

# TODO : typing causes cyclic import and import error, find solution for fixing Dataset import.
#from datum.datasets import Dataset
from datum.datasets import Entry, Observable
from datum.transformers import EntryMapper, ObservableMapper, StaticTransformer


class AttributesTransformer(StaticTransformer):
    """Class used to transform input dataset entries (and corresponding observable attributes)
    one by one by applying EntryMapper on some of their attributes subsets.

    Attributes:
        _entries_mappers (list): list of EntryMapper to apply to dataset entries.
        _observables_mappers (list): list of ObservableEntryMapper to apply to dataset
            observables.
    """
    def __init__(self, entries_mappers: Optional[List[EntryMapper]] = None,
                 observables_mappers: Optional[List[ObservableMapper]] = None):
        if entries_mappers:
            self._entries_mappers: List[EntryMapper] = entries_mappers
        else:
            self._entries_mappers: List[EntryMapper] = []
        if observables_mappers:
            self._observables_mappers: List[ObservableMapper] = observables_mappers
        else:
            self._observables_mappers: List[ObservableMapper] = []
        super().__init__()

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        return {'entries_mappers': [mapper.to_dict() for mapper in self._entries_mappers],
                'observables_mappers': [mapper.to_dict() for mapper in self._observables_mappers]}

    def transform(self, dataset, apply_mode: str='force') -> None:
        """Transform input dataset inplace by applying AttrubutesTransformer mappers to each of his
        entries and observables.

        Args:
            dataset: input dataset to transform.
            apply_mode: 'force' or 'optional'. 'optional' means that a mapper won't be
                applied to an entry/observable if it does not have required attributes
                (mapper in_attrs). With 'force', such a case will raise an Exception.
        """
        if apply_mode not in ['force', 'optional']:
            raise ValueError('apply_mode should be in |"force"|"optional"|')
        for entry, observables in dataset:
            # Apply entry mappers to entry
            for mapper in self._entries_mappers:
                mapper_out = mapper.apply_to(entry, apply_mode=apply_mode)
                if mapper_out:
                    dataset.update_entry_data(entry.idx, mapper_out)

            # Apply observables mappers to observables
            for obs in observables:
                for mapper in self._observables_mappers:
                    if mapper.obs_type != 'all' and mapper.obs_type != obs.typ:
                        continue
                    mapper_out = mapper.apply_to(obs, entry=entry, apply_mode=apply_mode)
                    if mapper_out:
                        dataset.update_observable_data(obs.idx, mapper_out)
