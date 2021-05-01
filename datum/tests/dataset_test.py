"""
Unit tests for Dataset class.
"""

import unittest

import pandas as pd

from datum.datasets import Entry, Observable, Dataset
from datum.utils.exceptions import MissingAttributes, MissingEntry, ForbiddenAttribute, EntryAlreadyExists


class TestEntry(unittest.TestCase):
    def test_init(self):
        entry1 = Entry({'width': 400, 'height': 400,
                        'name': 'image1', 'idx': 42, 'obs_ids': [1, 2]})
        self.assertEqual(entry1.name, 'image1')
        self.assertEqual(entry1.idx, 42)
        self.assertEqual(entry1.obs_ids, [1, 2])
        self.assertEqual(entry1['width'], 400)

        entry1 = Entry({'width': 400, 'height': 400},
                       name='image1', idx=42, obs_ids=[1, 2])
        self.assertEqual(entry1.name, 'image1')
        self.assertEqual(entry1.idx, 42)
        self.assertEqual(entry1.obs_ids, [1, 2])
        self.assertEqual(entry1['width'], 400)

    def test_add_obs(self):
        entry1 = Entry({'width': 400, 'height': 400,
                        'name': 'image1', 'idx': 42, 'obs_ids': [1, 2]})
        entry1._add_obs(3)
        self.assertEqual(entry1.obs_ids, [1, 2, 3])

    def test_remove_obs(self):
        entry1 = Entry({'width': 400, 'height': 400,
                        'name': 'image1', 'idx': 42, 'obs_ids': [1, 2]})
        entry1._remove_obs(2)
        self.assertEqual(entry1.obs_ids, [1])

    def test__update_data(self):
        entry1 = Entry({'width': 400, 'height': 400,
                        'name': 'image1', 'idx': 42, 'obs_ids': [1, 2]})
        entry1._update_data({'width': 600, 'depth': 3, 'name': 'image1.1'})
        self.assertEqual(entry1.name, 'image1.1')
        self.assertEqual(entry1['width'], 600)
        self.assertEqual(entry1['depth'], 3)

    def test_check_attributes(self):
        entry1 = Entry({'width': 400, 'height': 400,
                        'name': 'image1', 'idx': 42, 'obs_ids': [1, 2]})
        check = entry1.check_attributes(['width', 'name'])
        self.assertEqual(check, True)
        check = entry1.check_attributes(['width', 'depth'])
        self.assertEqual(check, False)

    def test_get(self):
        entry1 = Entry({'width': 400, 'height': 400,
                        'name': 'image1', 'idx': 42, 'obs_ids': [1, 2]})
        with self.assertRaises(MissingAttributes):
            entry1['depth']


class TestObservable(unittest.TestCase):
    def test_init(self):
        obs1 = Observable({'xmin': 0.0, 'name': 'car', 'type': 'object', 'entry_id': 42})
        self.assertEqual(obs1.typ, 'object')
        self.assertEqual(obs1.entry_id, 42)
        self.assertEqual(obs1['xmin'], 0.0)

        obs1 = Observable({'xmin': 0.0, 'name': 'car'}, typ='object', entry_id=42)
        self.assertEqual(obs1.typ, 'object')
        self.assertEqual(obs1.entry_id, 42)
        self.assertEqual(obs1['xmin'], 0.0)


class TestDataset(unittest.TestCase):
    def construct_basic_objects(self):
        entry1 = Entry({'width': 400, 'height': 400,
                        'name': 'image1'})
        entry2 = Entry({'width': 600, 'height': 600, 'name': 'image2',
                        'idx': 42, 'obs_ids': [1, 2]})
        obs1 = Observable({'xmin': 0.0, 'name': 'car', 'type': 'object'})
        obs2 = Observable({'xmin': 10.0, 'name': 'person', 'type': 'object',
                           'idx': 1254, 'entry_id': 42})
        dataset = Dataset(min_entries_attributes=['width'],
                          min_obs_common_attributes=['xmin'],
                          min_obs_type_attributes={'object': ['name']})
        return entry1, entry2, obs1, obs2, dataset

    def test_getitem(self):
        entry1, _, obs1, _, dataset = self.construct_basic_objects()
        dataset.add_entry(entry1, observables=[obs1])

        entry, _ = dataset['image1']
        self.assertEqual(entry.idx, 0)

        entry, _ = dataset[0]
        self.assertEqual(entry.name, 'image1')


    def test_add_entry(self):
        entry1, entry2, obs1, obs2, dataset = self.construct_basic_objects()

        dataset.add_entry(entry1, observables=[obs1])
        entry, obs = dataset['image1']
        self.assertEqual(entry.idx, 0)
        self.assertEqual(entry.name, 'image1')
        self.assertEqual(entry.obs_ids, [0])
        self.assertEqual(entry['width'], 400)
        self.assertEqual(obs[0].idx, 0)
        self.assertEqual(obs[0].entry_id, 0)
        self.assertEqual(obs[0]['xmin'], 0.0)

        dataset.add_entry(entry2, observables=[obs2])
        entry, obs = dataset['image2']
        self.assertEqual(entry.idx, 1)
        self.assertEqual(entry.name, 'image2')
        self.assertEqual(entry.obs_ids, [1])
        self.assertEqual(entry['width'], 600)
        self.assertEqual(obs[0].idx, 1)
        self.assertEqual(obs[0].entry_id, 1)
        self.assertEqual(obs[0]['xmin'], 10.0)

        entry3 = Entry({'height': 400,
                        'name': 'image3'})
        with self.assertRaises(MissingAttributes):
            dataset.add_entry(entry3)

        entry3 = Entry({'width': 400, 'height': 400,
                        'name': 'image1'})
        with self.assertRaises(EntryAlreadyExists):
            dataset.add_entry(entry3)


    def test_remove_entry(self):
        entry1, entry2, obs1, obs2, dataset = self.construct_basic_objects()
        dataset.add_entry(entry1, observables=[obs1])
        dataset.add_entry(entry2, observables=[obs2])
        dataset.remove_entry('image1')
        with self.assertRaises(MissingEntry):
            dataset.remove_entry('image1')

    def test_update_entry(self):
        entry1, entry2, obs1, obs2, dataset = self.construct_basic_objects()
        dataset.add_entry(entry1, observables=[obs1])
        dataset.add_entry(entry2, observables=[obs2])
        dataset.update_entry_data('image1', {'height': 1000, 'depth': 3, 'name': 'image1bis'})
        entry, _ = dataset['image1bis']
        self.assertEqual(entry['height'], 1000)
        self.assertEqual(entry['depth'], 3)

        with self.assertRaises(ForbiddenAttribute):
            dataset.update_entry_data('image1bis', {'idx': 10})

    def test_add_observable(self):
        entry1, _, obs1, obs2, dataset = self.construct_basic_objects()
        dataset.add_entry(entry1, observables=[obs1])
        dataset.add_observable('image1', obs2)
        entry, observables = dataset['image1']
        self.assertEqual(len(observables), 2)
        self.assertEqual(entry.obs_ids, [observables[0].idx, observables[1].idx])

        obs3 = Observable({'xmax': 0.0, 'name': 'car', 'type': 'object'})
        with self.assertRaises(MissingAttributes):
            dataset.add_observable('image1', obs3)

    def test_remove_observable(self):
        entry1, _, obs1, obs2, dataset = self.construct_basic_objects()
        dataset.add_entry(entry1, observables=[obs1, obs2])
        entry, [obs1, obs2] = dataset['image1']
        dataset.remove_observable(obs1.idx)
        self.assertEqual(entry.obs_ids, [obs2.idx])
        with self.assertRaises(MissingEntry):
            dataset.remove_observable(obs1.idx)

    def test_update_observable(self):
        entry1, _, obs1, obs2, dataset = self.construct_basic_objects()
        dataset.add_entry(entry1, observables=[obs1, obs2])
        entry, observables = dataset['image1']
        dataset.update_observable_data(observables[0].idx,
                                       {'xmin': 0.0, 'name': 'car', 'type': 'object'})
        self.assertEqual(len(observables), 2)
        self.assertEqual(entry.obs_ids, [observables[0].idx, observables[1].idx])

        with self.assertRaises(ForbiddenAttribute):
            dataset.update_observable_data(observables[0].idx, {'idx': 42})

        with self.assertRaises(ForbiddenAttribute):
            dataset.update_observable_data(observables[0].idx, {'entry_id': 42})

        with self.assertRaises(ForbiddenAttribute):
            dataset.update_observable_data(observables[0].idx, {'type': 'segmentation'})

    def test_add(self):
        entry1 = Entry({'width': 400, 'height': 400, 'name': 'image1'})
        obs1 = Observable({'xmin': 0.0, 'name': 'car', 'type': 'object'})
        dataset1 = Dataset(min_entries_attributes=['width'],
                           min_obs_common_attributes=['xmin'],
                           min_obs_type_attributes={'object': ['name']})
        dataset1.add_entry(entry1, observables=[obs1])

        entry2 = Entry({'width': 600, 'height': 600, 'name': 'image2'})
        obs2 = Observable({'xmin': 10.0, 'ymin': 10.0, 'name': 'person',
                           'other': 0,'type': 'object'})
        dataset2 = Dataset(min_entries_attributes=['width', 'height'],
                           min_obs_common_attributes=['xmin', 'ymin'],
                           min_obs_type_attributes={'object': ['name', 'other']})
        dataset2.add_entry(entry2, observables=[obs2])

        dataset = dataset1 + dataset2
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.min_entries_attributes, dataset1.min_entries_attributes)
        self.assertEqual(dataset.min_obs_common_attributes, dataset1.min_obs_common_attributes)
        self.assertEqual(dataset.min_obs_type_attributes, dataset1.min_obs_type_attributes)

    def test_merge(self):
        entry1 = Entry({'width': 400, 'height': 400, 'name': 'image1'})
        obs1 = Observable({'xmin': 0.0, 'name': 'car', 'type': 'object'})
        dataset1 = Dataset(min_entries_attributes=['width'],
                           min_obs_common_attributes=['xmin'],
                           min_obs_type_attributes={'object': ['name']})
        dataset1.add_entry(entry1, observables=[obs1])

        entry2 = Entry({'width': 600, 'height': 600, 'name': 'image2'})
        obs2 = Observable({'xmin': 10.0, 'ymin': 10.0, 'name': 'person',
                           'other': 0,'type': 'object'})
        dataset2 = Dataset(min_entries_attributes=['width', 'height'],
                           min_obs_common_attributes=['xmin', 'ymin'],
                           min_obs_type_attributes={'object': ['name', 'other']})
        dataset2.add_entry(entry2, observables=[obs2])

        dataset = Dataset.merge([dataset1, dataset2], attributes_merging='intersection')
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.min_entries_attributes, ['width'])
        self.assertEqual(dataset.min_obs_common_attributes, ['xmin'])
        self.assertEqual(dataset.min_obs_type_attributes, {'object': ['name']})

        with self.assertRaises(MissingAttributes):
            dataset = Dataset.merge([dataset1, dataset2], attributes_merging='union')

    def test_from_dataframes(self):
        entries = {0: {'idx': 0, 'name': 'image1', 'width': 400},
                   1: {'idx': 1, 'name': 'image2', 'width': 600}}
        entries = pd.DataFrame.from_dict(entries, orient='index')
        obs = {0: {'idx': 0, 'entry_id': 0, 'name': 'dog', 'xmin': 0.0, 'type': 'object'},
               1: {'idx': 1, 'entry_id': 1, 'name': 'person', 'xmax': 1.0, 'type': 'box'}}
        obs = pd.DataFrame.from_dict(obs, orient='index')

        dataset = Dataset.from_dataframes(entries, obs)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.min_entries_attributes, ['width'])
        self.assertEqual(dataset.min_obs_common_attributes, ['name'])
        # TODO : make this test pass
        #self.assertEqual(dataset.min_obs_type_attributes, {'object': ['xmin'], 'box': ['xmax']})

    def test_remove_entries_without_obs(self):
        entry1, entry2, obs1, obs2, dataset = self.construct_basic_objects()
        dataset.add_entry(entry1, observables=[obs1])
        dataset.add_entry(entry2, observables=[obs2])
        _, obs = dataset['image1']
        dataset.remove_observable(obs[0].idx)
        dataset.remove_entries_without_obs()
        self.assertEqual(len(dataset), 1)

if __name__ == '__main__':
    unittest.main()
