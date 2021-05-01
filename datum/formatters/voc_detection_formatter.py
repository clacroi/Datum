"""
Generic module to format and save a Dataset object under VOC-Detection format.
VOC-Detection format is defined by a set of minimal construction rules :
- images are stored under <root_dir>/JPEGImages/
- 1 XML annotation file per image is used to store image annotations (image attributes)
  and corresponding objects annotations (observables of type 'object')
- annotations files are stored under <root_dir>/Annotations/
- annotations files contain an root node <annotation>
- annotations files contain a node <size> containing subnodes <height> and <width>
  denoting images dimensions
- annotations files contain 1 <object> node per object with sudnodes
-- <name> : the name of the class of the object
-- <bndbox> : <xmin>, <ymin>, <xmax>, <ymax> denote the coordinates in image
   of the top-left and bottom-right corners of object bounding box. Coordinates
   are in [1, width/height]
"""

import os
from shutil import copy2
from typing import Tuple
from pathlib import Path
from typing import Union, List, Dict, Any, Optional

from datum.datasets import Dataset, Entry
from datum.formatters import DatasetFormatter
from datum.utils.io_utils import dump_xml_annotation, dump_list_to_file
from datum.utils.annotations_utils import set_dict_value, remove_file_extension
from datum.utils.exceptions import MissingAttributes, ImpossibleAnnotationExtraction


class VocAnnotationConstructor():
    """Simple class for representing a VOC single annotation and more specifically its construction
    rules: its name, its path in XML tree structure, value type, default value, if its value should
    be set to its default value if the annotation can't be constructed from a dataset entry.
    This class does not represent the value of the annotation itself.

    Attributes:
        _name: annotation corresponding attribute name.
        _path: path of annotation in XML tree-like.
        _typ (Python type): the annotation value type.
        _default_val (_typ): the annotation default value.
        _set_to_default: whether annotation value should be set to default if annotation.
        can't be extracted.
    """
    def __init__(self, name: str, path: List, typ: type,
                  default_val: Any, set_to_default: bool):
        self._name: str = name
        self._path: List = path
        self._typ: type = typ
        self._default_val = default_val
        self._set_to_default: bool = set_to_default

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        return {'constructor_type': type(self).__name__,
                'name': self._name,
                'path': self._path,
                'typ': self._typ,
                'default_val': self._default_val,
                'set_to_default': self._set_to_default}

    @property
    def name(self):
        return self._name
    @property
    def path(self):
        return self._path
    @property
    def typ(self):
        return self._typ
    @property
    def set_to_default(self):
        return self._set_to_default
    @property
    def default_val(self):
        return self._default_val

    def extract_in(self, entry: Union[Entry, Dict[str, Any]]) -> Tuple[bool, Optional[Any]]:
        """Extracts value of the annotation in flat data dictionary of entry/observable attributes.

        Args:
            entry: input flat attributes dictionary.
        Returns:
            bool indicating whether extraction executed correctly.
            extracted value for the annotation.
        """
        if self.name not in entry:
            if self.set_to_default:
                return True, self.default_val
            else:
                return False, None
        else:
            try:
                val = self.typ(entry[self.name])
            except ValueError:
                return False, None
        return True, val

    def __str__(self):
        return 'VocAnnotationConstructor : (name="{}", path={}, type={}, default_val={}, ' \
               'set_to_default={})'.format(self.name, self.path, self.typ,
                                           self.default_val, self.set_to_default)

class VocObsAnnotationConstructor(VocAnnotationConstructor):
    """Simple class for representing a VOC single observable annotation and more specifically its
    construction rules: the type of observable it is related to, its name, value type,
    default value, if its value should be set to its default value if the annotation value
    can't be extracted from input entry/observable attributes.
    This class does not represent the value of the attribute itself.

    Attributes:
        _observable_type: observable type the annotation is related to.
    """
    def __init__(self, observable_type, name, path,
                 typ, default_val, set_to_default):
        super().__init__(name, path, typ, default_val, set_to_default)
        self._observable_type: str = observable_type

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        return {'constructor_type': type(self).__name__,
                'name': self._name,
                'path': self._path,
                'typ': self._typ,
                'default_val': self._default_val,
                'set_to_default': self._set_to_default,
                'observable_type': self._observable_type}

    @property
    def observable_type(self):
        return self._observable_type

class VocDetectionFormatter(DatasetFormatter):
    """Class to save a Dataset object under the VOC-Detection format.

    Attributes:
        _images_dir: path to VOC images folder.
        _annotations_dir: path to VOC annotations files folder.
        _images_annotations: list of images annotations.
        _obs_annotations: list of observables annotations.
    """
    # File extension for VOC-like database annotations
    ANN_EXT = 'xml'
    # File extension for VOC-like database images
    IMG_EXT = 'jpg' # TODO : pass it as class init arg

    # Default images annotations (can be overriden by input image_annotations)
    IMAGES_ANNOTATIONS = [VocAnnotationConstructor('width', ['annotation', 'size', 'width'],
                                                   int, None, False),
                          VocAnnotationConstructor('height', ['annotation', 'size', 'height'],
                                                   int, None, False),
                          VocAnnotationConstructor('depth', ['annotation', 'size', 'depth'],
                                                   int, 3, True)]
    # Default observables annotations construction rules (can be overiden by input obs_annotations)
    OBS_ANNOTATIONS = [VocObsAnnotationConstructor('object', 'name', ['name'],
                                                   str, None, False),
                       VocObsAnnotationConstructor('object', 'xmin', ['bndbox', 'xmin'],
                                                   int, None, False),
                       VocObsAnnotationConstructor('object', 'ymin', ['bndbox', 'ymin'],
                                                   int, None, False),
                       VocObsAnnotationConstructor('object', 'xmax', ['bndbox', 'xmax'],
                                                   int, None, False),
                       VocObsAnnotationConstructor('object', 'ymax', ['bndbox', 'ymax'],
                                                   int, None, False)]

    def __init__(self, root_dir: Path,
                 images_annotations=None,
                 obs_annotations=None) -> None:
        super().__init__(root_dir)
        self._images_dir: Path = root_dir / 'JPEGImages'
        self._annotations_dir: Path = root_dir / 'Annotations'
        self._imagesets_dir: Path = root_dir / 'ImageSets' / 'Main'

        # Construct images annotations to format
        if images_annotations:
            self._images_annotations: List = images_annotations
        else:
            self._images_annotations: List = []
        for default_ann in self.IMAGES_ANNOTATIONS:
            to_add = True
            for ann in self._images_annotations:
                if default_ann.name == ann.name:
                    to_add = False # default attribute is replaced by user input attribute
                    break
            if to_add:
                self._images_annotations.append(default_ann)
        for ann in self._images_annotations:
            print('Constructing annotation "{}" with path {}, type {}, '
                  'default value={} and set_to_default={}'
                  .format(ann.name, ann.path, ann.typ, ann.default_val, ann.set_to_default))

        # Construct observables annotations to format
        if obs_annotations:
            self._obs_annotations: List = obs_annotations
        else:
            self._obs_annotations: List = []
        for default_ann in self.OBS_ANNOTATIONS:
            to_add = True
            for ann in self._obs_annotations:
                if (default_ann.observable_type == ann.observable_type) \
                        and (default_ann.name == ann.name):
                    to_add = False # default annotation is replaced by user input annotation
                    break
            if to_add:
                self._obs_annotations.append(default_ann)

        for ann in self._obs_annotations:
            print('Constructing annotation "{}" for observable type "{}" with path {}, '
                  'type {}, default value={} and set_to_default={}'
                    .format(ann.name, ann.observable_type, ann.path, ann.typ, ann.default_val,
                            ann.set_to_default))

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        return {'formatter_type': type(self).__name__,
                'root_dir': str(self.root_dir),
                'images_dir': str(self._images_dir),
                'annotations_dir': str(self._annotations_dir),
                'imagesets_dir': str(self._imagesets_dir),
                'images_annotations': [ann.to_dict() for ann in self._images_annotations],
                'obs_annotations': [ann.to_dict() for ann in self._obs_annotations]}

    def format(self, dataset: Dataset, copy_images=True) -> None:
        """Saves input dataset under VOC-Detection format.
        Args:
            dataset: dataset object format and save.
            copy_images: whether to save or not images on disk.
        """
        # Check that all entries from dataset have valid value for colums 'dir' and 'filename'
        images_df = dataset.entries_df
        for col in ['dir', 'filename', 'set']:
            if col not in images_df.columns:
                raise MissingAttributes(dataset, col)
            if images_df[col].isnull().any():
                raise MissingAttributes(dataset, col)

        if not self._images_dir.exists():
            os.makedirs(self._images_dir)
        if not self._annotations_dir.exists():
            os.makedirs(self._annotations_dir)
        if not self._imagesets_dir.exists():
            os.makedirs(self._imagesets_dir)

        for image, observables in dataset:
            image_name = image.name
            annotations = {}
            # Construct image annotations
            for ann in self._images_annotations:
                valid, ann_val = ann.extract_in(image)
                if not valid:
                    raise ImpossibleAnnotationExtraction(ann, image)
                set_dict_value(annotations, ann.path, ann_val)

            # Construct observables annotations
            for obs in observables:
                obs_annotation = {}
                for ann in self._obs_annotations:
                    if obs.typ != ann.observable_type:
                        continue
                    valid, ann_val = ann.extract_in(obs)
                    if not valid:
                        raise ImpossibleAnnotationExtraction(ann, obs)
                    set_dict_value(obs_annotation, ann.path, ann_val)
                if obs.typ not in annotations['annotation']:
                    annotations['annotation'][obs.typ] = [obs_annotation]
                else:
                    annotations['annotation'][obs.typ].append(obs_annotation)

            # Write annotation file and image
            dump_xml_annotation(annotations,
                                self._annotations_dir / (image_name + '.' + self.ANN_EXT))
            if copy_images:
                src_image_file = Path(image['dir'] / image['filename'])
                if not src_image_file.exists():
                    raise Exception('Image file {} does not exist'.format(src_image_file))
                dst_img_file = Path(self._images_dir / (image_name + '.' + self.IMG_EXT))
                copy2(src_image_file, dst_img_file)

        # Write images set lists
        images_df['list_name'] = images_df['filename'].apply(lambda x: remove_file_extension(x))
        images_sets = images_df['set'].unique()
        for images_set in images_sets:
            images = images_df[images_df['set'] == images_set]['list_name'].values
            dump_list_to_file(images, self._imagesets_dir / '{}.txt'.format(images_set))
