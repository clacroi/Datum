"""
Generic module to populate a Dataset object from a COCO-formatted detection database.

Use example : see '__main__' section.
"""

from tqdm import tqdm
from pathlib import Path
from typing import List, Set, Dict, Any, Optional
import json

from bidict import bidict

from datum.datasets import Dataset
from datum.readers import DatasetReader, AttributeConstructor, ObsAttributeConstructor
from datum.readers import update_obs_attr_constructors, update_entries_attr_constructors
from datum.utils.annotations_utils import remove_file_extension
from datum.utils.exceptions import ImpossibleAttributeExtraction


class CocoDetectionReader(DatasetReader):
    """Class to populate a Dataset object from a VOC-like detection database.

    Attributes:
        img_ext: images files extenstion.
        _entries_constructors: list of attributes to extract for each image
        _obs_constructors: list of observable attributes to extract for each
        observable.
    """
    # Coco sets
    COCO_SETS = ['train2017', 'val2017', 'train2014', 'val2014']
    COCO_LABELMAP = bidict({0: "background", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
                            5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat",
                            10: "traffic light", 11: "fire hydrant", 13: "stop sign",
                            14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog",
                            19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
                            24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella",
                            31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis",
                            36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
                            40: "baseball glove", 41: "skateboard", 42: "surfboard",
                            43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup",
                            48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
                            53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
                            57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
                            62: "chair", 63: "couch", 64: "potted plant", 65: "bed",
                            67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse",
                            75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
                            79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
                            85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
                            89: "hair drier", 90: "toothbrush"})

    # File extension for VOC-like database annotations
    ANN_EXT = '.json'

    # Default dataset entries and observables common/per-type attributes to extract
    DEFAULT_ENTRIES_CONSTRUCTORS = [AttributeConstructor('filename', 'custom', str, None, False),
                                    AttributeConstructor('dir', 'custom', Path, None, False),
                                    AttributeConstructor('width', 'lookup', int, None, False,
                                                         ['width']),
                                    AttributeConstructor('height', 'lookup', int, None, False,
                                                         ['height']),
                                    AttributeConstructor('depth', 'lookup', int, 3, True,
                                                         ['depth']),
                                    AttributeConstructor('set', 'custom', str, None, False)]
    DEFAULT_OBS_CONSTRUCTORS = [ObsAttributeConstructor('object', 'name', 'custom',
                                                        str, None, False),
                                ObsAttributeConstructor('object', 'xmin', 'custom',
                                                        float, None, False),
                                ObsAttributeConstructor('object', 'ymin', 'custom',
                                                        float, None, False),
                                ObsAttributeConstructor('object', 'xmax', 'custom',
                                                        float, None, False),
                                ObsAttributeConstructor('object', 'ymax', 'custom',
                                                        float, None, False)]

    def __init__(self, root_dir: Path, img_ext: str,
                 entries_constructors: Optional[List[AttributeConstructor]] = None,
                 obs_constructors: Optional[List[ObsAttributeConstructor]] = None) -> None:
        if entries_constructors:
            coco_entries_constructors = \
                update_obs_attr_constructors(self.DEFAULT_ENTRIES_CONSTRUCTORS,
                                             entries_constructors)
        else:
            coco_entries_constructors = self.DEFAULT_ENTRIES_CONSTRUCTORS

        if obs_constructors:
            coco_obs_constructors = \
                update_obs_attr_constructors(self.DEFAULT_OBS_CONSTRUCTORS, obs_constructors)
        else:
            coco_obs_constructors = self.DEFAULT_OBS_CONSTRUCTORS
        super().__init__(root_dir,
                         entries_constructors=coco_entries_constructors,
                         obs_constructors=coco_obs_constructors)
        self.img_ext: str = img_ext

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        return {'reader_type': type(self).__name__,
                'root_dir': str(self.root_dir),
                'entries_constructors': [const.to_dict() for const in self._entries_constructors],
                'obs_constructors': [const.to_dict() for const in self._obs_constructors],
                'img_ext': self.img_ext}

    def feed(self, dataset: Dataset,
             sets: List[str] = ['train2017', 'val2017'],
             clear_existing_data: bool=False) -> None:
        """Extracts all COCO individual entries (images) attributes and their corresponding
        observables (objects, ..) attributes and populates input dataset with.
        Args:
            dataset: dataset object to populate.
            sets: sets of images to load.
            clear_existing_data: whether to clean existing entries in dataset.
        """
        annotations_files = []
        for coco_set in sets:
            if coco_set not in self.COCO_SETS:
                raise ValueError
            annotations_files.append(self.root_dir / 'annotations' /
                                     ('instances_' + coco_set + self.ANN_EXT))

        if clear_existing_data:
            dataset.clear_data()

        for coco_set, annotation_file in zip(sets, annotations_files):
            data = json.load(open(annotation_file))
            id_to_name = {}
            # Construct entries attributes and add entries (images) to dataset
            for image in tqdm(data["images"]):
                name = remove_file_extension(image['file_name'])
                image_attributes = {'name': name}
                for constructor in self._entries_constructors:
                    if constructor.process == 'lookup':
                        valid, attr_val = constructor.extract_in(image)
                        if not valid:
                            raise ImpossibleAttributeExtraction(constructor, image)
                        image_attributes[constructor.name] = attr_val
                    else:
                        if constructor.name == 'filename':
                            image_attributes[constructor.name] = image['file_name']
                        elif constructor.name == 'dir':
                            image_attributes[constructor.name] = self.root_dir / coco_set
                        elif constructor.name == 'set':
                            image_attributes[constructor.name] = coco_set
                        else:
                            raise NotImplementedError
                dataset.add_entry(image_attributes)
                id_to_name[image['id']] = name

            # Construct objects attributes and add object observables to dataset
            for obs in tqdm(data["annotations"]):
                obs_attributes = {'type': 'object'}
                xmin, ymin, width, height = obs['bbox']
                xmax, ymax = xmin + width, ymin + height
                image_name = id_to_name[obs['image_id']]
                for constructor in self._obs_constructors:
                    if constructor.observable_type != 'object':
                        continue
                    if constructor.process == 'lookup':
                        valid, attr_val = constructor.extract_in(obs)
                        if not valid:
                            raise ImpossibleAttributeExtraction(constructor, obs)
                        obs_attributes[constructor.name] = attr_val
                    else:
                        if constructor.name == 'name':
                            obs_attributes[constructor.name] = \
                                self.COCO_LABELMAP[obs["category_id"]]
                        elif constructor.name == 'xmin':
                            obs_attributes[constructor.name] = constructor.typ(xmin)
                        elif constructor.name == 'ymin':
                            obs_attributes[constructor.name] = constructor.typ(ymin)
                        elif constructor.name == 'xmax':
                            obs_attributes[constructor.name] = constructor.typ(xmax)
                        elif constructor.name == 'ymax':
                            obs_attributes[constructor.name] = constructor.typ(ymax)
                        else:
                            raise NotImplementedError
                # Add observable to dataset
                dataset.add_observable(image_name, obs_attributes)

