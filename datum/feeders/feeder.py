"""
Generic module to represent a feeder, i.e an object responsible for constructing examples of data
from a input dataset. It may for example construct batch of data for feeding a neural network
during training.
"""

import random
from typing import Optional, List, Dict, Tuple, Any

from datum.datasets import Dataset, Entry, Observable


class Feeder():
    """Class to represent a generic feeder.

    Attributes:
        _dataset: dataset to take data from.
        batch_size: size of data batches.
        shuffle: whether data examples should be taken from input dataset randomly.
        _seed: feeder random seed.
        force_batch_size: whether batches of data should always have batch_size or not.
        _processed_entries_idxs: list of entries indexes from input dataset to process.
        _to_process_entries_idxs: list of already processed entries indexes from input dataset.
    """
    def __init__(self, dataset: Dataset, batch_size: int,
                 shuffle: Optional[bool] = False,
                 seed: Optional[int] = 42,
                 force_batch_size: Optional[bool] = True) -> None:
        self._dataset: Dataset = dataset
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self._seed: int = seed
        random.seed(self._seed)
        self.force_batch_size: bool = force_batch_size

        self._processed_entries_idxs: List[int] = []
        self._to_process_entries_idxs: List[int] = self._dataset.entries_idxs_list

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        return {'feeder_type': type(self).__name__,
                'dataset': self._dataset.to_dict(),
                'shuffle': self.shuffle,
                'seed': self._seed,
                'force_batch_size': self.force_batch_size}

    def _process_example(self, entry: Entry,
                         observables: List[Observable]) -> Tuple[Entry, List[Observable]]:
        """Process a single example from dataset, i.e a entry + its observables.

        Args:
            entry: input entry to process.
            observables: entry related observables to process.

        Returns
            processed entry and observables.
        """
        return entry, observables

    def _process_batch(self, batch: List[Tuple[Entry, List[Observable]]]) \
            -> List[Tuple[Entry, List[Observable]]]:
        """Process a single batch of data, i.e a list of entries + corresponding observables
        from _dataset.

        Args:
            batch: input batch to process.

        Returns
            processed entries and observables as batch of data.
        """
        return batch

    def reset_state(self) -> None:
        """Reset feeder internal state / data structures."""
        self._processed_entries_idxs = []
        self._to_process_entries_idxs = self._dataset.entries_idxs_list

    def _get_next_example_idx(self) -> int:
        """Get id of next entry from _dataset to process.

        Returns:
            entry id
        """
        if self.shuffle:
            return random.randint(0, min(self.batch_size, len(self._to_process_entries_idxs) - 1))
        else:
            return 0

    def __next__(self) -> List[Tuple[Entry, List[Observable]]]:
        """Compute and return next batch of data from _dataset.

        Returns:
            batch
        """
        if len(self._to_process_entries_idxs) == 0 or \
                    (len(self._to_process_entries_idxs) < self.batch_size and self.force_batch_size):
            self.reset_state()
        data_batch = []
        next_batch_size = min(self.batch_size, len(self._to_process_entries_idxs))
        for _ in range(next_batch_size):
            next_example_idx = self._get_next_example_idx()
            entry, observables = self._dataset[self._to_process_entries_idxs[next_example_idx]]
            data_example = self._process_example(entry, observables)
            data_batch.append((data_example))
            self._processed_entries_idxs.append(self._to_process_entries_idxs[next_example_idx])
            self._to_process_entries_idxs.pop(next_example_idx)
        return self._process_batch(data_batch)

    def iter(self, n_iter: int) -> List[Tuple[Entry, List[Observable]]]:
        """Iterator over n_iter batches of data.

        Returns:
            batch
        """
        self.reset_state()
        for _ in range(n_iter):
            yield self.__next__()
