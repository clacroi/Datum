"""
Unit tests for reader utility classes and functions.
"""

import unittest

from datum.readers import AttributeConstructor, ObsAttributeConstructor
from datum.utils.exceptions import MissingAttributes, MissingEntry, ForbiddenAttribute, EntryAlreadyExists


class TestAttributeConstructor(unittest.TestCase):
    def test_extract_in(self):
        const = AttributeConstructor('width', 'lookup', int, None, False,
                                     lookup_path=['size', 'width'])
        data = {'size': {'height': 400, 'width': 600}}
        valid, val = const.extract_in(data)
        self.assertEqual(valid, True)
        self.assertEqual(val, 600)

        data = {'size': {'height': 400, 'width': '600'}}
        valid, val = const.extract_in(data)
        self.assertEqual(valid, True)
        self.assertEqual(val, 600)

        data = {'size': {'height': 400}}
        valid, val = const.extract_in(data)
        self.assertEqual(valid, False)
        self.assertEqual(val, None)

        const = AttributeConstructor('width', 'lookup', int, 0, True,
                                     lookup_path=['size', 'width'])
        data = {'size': {'height': 400}}
        valid, val = const.extract_in(data)
        self.assertEqual(valid, True)
        self.assertEqual(val, 0)

        const = AttributeConstructor('width', 'lookup', int, None, False,
                                     lookup_path=['size', 'width'])
        data = {'size': {'height': 400, 'width': 'test'}}
        valid, val = const.extract_in(data)
        self.assertEqual(valid, False)
        self.assertEqual(val, 'test')

if __name__ == '__main__':
    unittest.main()
