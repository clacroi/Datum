"""
Datum exceptions definitions.
"""

class MissingAttributes(Exception):
    def __init__(self, obj, attribute_names=None):
        if attribute_names is not None:
            self.message = 'Missing attributes {} in {}'.format(attribute_names, obj)
        else:
            self.message = 'Missing attributes in {}'.format(obj)
    def __str__(self):
        return self.message

class ForbiddenAttribute(Exception):
    def __init__(self, obj, attribute_name=None):
        if attribute_name is not None:
            self.message = 'Modification of forbidden attribute {} in {}'.format(attribute_name, obj)
        else:
            self.message = 'Modification of forbidded attribute in {}'.format(obj)
    def __str__(self):
        return self.message

class MissingEntry(Exception):
    def __init__(self, token, entry_type='Entry'):
        self.message = 'Missing entry of type {} with token {}'.format(entry_type, token)
    def __str__(self):
        return self.message

class EntryAlreadyExists(Exception):
    def __init__(self, token, entry_type='Entry'):
        self.message = 'Entry of type {} with token {} already exists'.format(entry_type, token)
    def __str__(self):
        return self.message

class ImpossibleAttributeExtraction(Exception):
    def __init__(self, constructor, data):
        self.message = 'Impossible to use constructor {} in data {}'.format(constructor, data)
    def __str__(self):
        return self.message

class ImpossibleAnnotationExtraction(Exception):
    def __init__(self, constructor, entry):
        self.message = 'Impossible to apply annotation constructor {} on entry {}'\
            .format(constructor, entry)
    def __str__(self):
        return self.message
