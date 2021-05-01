"""
Unit tests for transformer utility classes and functions.
"""

import unittest

from datum.datasets import Entry, Observable, Dataset
from datum.readers import AttributeConstructor, ObsAttributeConstructor
from datum.utils.exceptions import MissingAttributes, ForbiddenAttribute
from datum.transformers import EntryMapper, ObservableMapper, AttributesTransformer


class TestTransformer(unittest.TestCase):
    def construct_basic_objects(self):
        entry1 = Entry({'width': 400, 'height': 400, 'name': 'image1'})
        entry2 = Entry({'width': 600, 'height': 600, 'name': 'image2'})
        obs1 = Observable({'xmin': 10.0, 'xmax': 20.0, 'name': 'car', 'type': 'box'})
        obs2 = Observable({'xmin': 20.0, 'xmax': 40.0, 'name': 'person', 'type': 'object'})
        dataset = Dataset(min_entries_attributes=['width'],
                          min_obs_common_attributes=['xmin'],
                          min_obs_type_attributes={'object': ['name']})

        return entry1, entry2, obs1, obs2, dataset

    def test_transform(self):
        entry1, entry2, obs1, obs2, dataset = self.construct_basic_objects()
        dataset.add_entry(entry1, observables=[obs1])
        dataset.add_entry(entry2, observables=[obs2])

        width_divider = EntryMapper(['width'], ['width'], lambda x: x / 2.0)
        transformer = AttributesTransformer(entries_mappers=[width_divider])
        transformer.transform(dataset)
        self.assertEqual(dataset['image1'][0]['width'], 200.0)
        self.assertEqual(dataset['image2'][0]['width'], 300.0)

        depth_divider = EntryMapper(['depth'], ['depth'], lambda x: x / 2.0)
        transformer = AttributesTransformer(entries_mappers=[depth_divider])
        with self.assertRaises(MissingAttributes):
            transformer.transform(dataset)

        id_modifier = EntryMapper(['idx'], ['idx'], lambda x: x * 2)
        transformer = AttributesTransformer(entries_mappers=[id_modifier])
        with self.assertRaises(ForbiddenAttribute):
            transformer.transform(dataset)

        xmin_mapper = ObservableMapper(['xmin'], ['xmin'], lambda x: x -1.0, obs_type='all')
        transformer = AttributesTransformer(observables_mappers=[xmin_mapper])
        transformer.transform(dataset)
        self.assertEqual(dataset['image1'][1][0]['xmin'], 9.0)
        self.assertEqual(dataset['image2'][1][0]['xmin'], 19.0)

        xmin_mapper = ObservableMapper(['xmin'], ['xmin'], lambda x: x -1.0, obs_type='object')
        transformer = AttributesTransformer(observables_mappers=[xmin_mapper])
        transformer.transform(dataset)
        self.assertEqual(dataset['image1'][1][0]['xmin'], 9.0)
        self.assertEqual(dataset['image2'][1][0]['xmin'], 18.0)

        xmin_mapper = ObservableMapper(['xmin'], ['xmin'], lambda x: x -1.0, obs_type='box')
        transformer = AttributesTransformer(observables_mappers=[xmin_mapper])
        transformer.transform(dataset)
        self.assertEqual(dataset['image1'][1][0]['xmin'], 8.0)
        self.assertEqual(dataset['image2'][1][0]['xmin'], 18.0)

        entry_id_modifier = ObservableMapper(['entry_id'], ['entry_id'], lambda x: x * 2)
        transformer = AttributesTransformer(observables_mappers=[entry_id_modifier])
        with self.assertRaises(ForbiddenAttribute):
            transformer.transform(dataset)


if __name__ == '__main__':
    unittest.main()
