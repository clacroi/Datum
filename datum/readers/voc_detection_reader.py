"""
Generic module to populate a Dataset object from a VOC-formatted detection database.

For each image, some attributes are extracted from corresponding file.
- _entries_constructors is a list of constructors, each extracting 1 attribute in each image.
- _obs_constructors is a list of constructors, each extracting 1 attribute in each observable
  with the same type of the attribute.

Use example :

from pathlib import Path
from python.datasets.readers.voc_detection_reader import VocDetectionReader
from python.datasets.readers.reader import ObsAttributeConstructor

root_dir = Path(<voc-like_data_root>)
test_images_list = Path(<images_list>)
pose_attribute = ObsAttributeConstructor('object', 'pose', 'lookup', str, 'unknown', True, ['pose'])
reader = VocDetectionReader(root_dir, 'jpg',
                            set_images_lists={'test': test_images_list},
                            entries_constructors=[],
                            obs_constructors=[pose_attribute])
dataset = Dataset()
reader.feed(dataset)
"""

from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Any

from datum.datasets import Dataset
from datum.readers import DatasetReader, AttributeConstructor, ObsAttributeConstructor
from datum.readers import update_obs_attr_constructors, update_entries_attr_constructors
from datum.utils.io_utils import read_list_file, load_xml_annotation
from datum.utils.annotations_utils import remove_file_extension
from datum.utils.exceptions import ImpossibleAttributeExtraction


class VocDetectionReader(DatasetReader):
    """Class to populate a Dataset object from a VOC-like detection database.

    Attributes:
        img_ext: image files extension.
        set_images_list: dictionary mapping images set to their list of images.
        _entries_constructors: list of attributes constructors for each image.
        _obs_constructors: list of observable constructors.
    """
    # File extension for VOC-like database annotations
    ANN_EXT = '.xml'

    # Default dataset entries and observables attributes to extract
    DEFAULT_ENTRIES_CONSTRUCTORS = [AttributeConstructor('filename', 'custom', str, None, False),
                                    AttributeConstructor('dir', 'custom', Path, None, False),
                                    AttributeConstructor('width', 'lookup', int, None, False,
                                                         ['annotation', 'size', 'width']),
                                    AttributeConstructor('height', 'lookup', int, None, False,
                                                         ['annotation', 'size', 'height']),
                                    AttributeConstructor('set', 'custom', str, 'unknown', False)]

    DEFAULT_OBS_CONSTRUCTORS = [ObsAttributeConstructor('object', 'name', 'lookup',
                                                        str, None, False, ['name']),
                                ObsAttributeConstructor('object', 'xmin', 'lookup',
                                                        float, None, False, ['bndbox', 'xmin']),
                                ObsAttributeConstructor('object', 'ymin', 'lookup',
                                                        float, None, False, ['bndbox', 'ymin']),
                                ObsAttributeConstructor('object', 'xmax', 'lookup',
                                                        float, None, False, ['bndbox', 'xmax']),
                                ObsAttributeConstructor('object', 'ymax', 'lookup',
                                                        float, None, False, ['bndbox', 'ymax'])]

    def __init__(self, root_dir: Path, img_ext: str,
                 set_images_lists: Dict = {},
                 annotations_dir: Optional[Path] = None,
                 images_dir: Optional[Path] = None,
                 entries_constructors: Optional[List[AttributeConstructor]] = None,
                 obs_constructors: Optional[List[ObsAttributeConstructor]] = None) -> None:
        if entries_constructors:
            voc_entries_constructors = \
                update_obs_attr_constructors(self.DEFAULT_ENTRIES_CONSTRUCTORS,
                                                   entries_constructors)
        else:
            voc_entries_constructors = self.DEFAULT_ENTRIES_CONSTRUCTORS

        if obs_constructors:
            voc_obs_constructors = \
                update_obs_attr_constructors(self.DEFAULT_OBS_CONSTRUCTORS, obs_constructors)
        else:
            voc_obs_constructors = self.DEFAULT_OBS_CONSTRUCTORS

        super().__init__(root_dir,
                         entries_constructors=voc_entries_constructors,
                         obs_constructors=voc_obs_constructors)
        self.img_ext: str = img_ext
        self.set_images_lists: Dict = set_images_lists
        if annotations_dir:
            self.annotations_dir: Path = annotations_dir
        else:
            self.annotations_dir: Path = self.root_dir / 'Annotations'
        if images_dir:
            self.images_dir: Path = images_dir
        else:
            self.images_dir: Path = self.root_dir / 'JPEGImages'

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        return {'reader_type': type(self).__name__,
                'root_dir': str(self.root_dir),
                'entries_constructors': [const.to_dict() for const in self._entries_constructors],
                'obs_constructors': [const.to_dict() for const in self._obs_constructors],
                'img_ext': self.img_ext,
                'set_images_lists': {set_name: str(set_list) \
                    for (set_name, set_list) in self.set_images_lists.items()}}

    def feed(self, dataset: Dataset, clear_existing_data: bool=False) -> None:
        """Extracts all VOC individual entries (images) attributes and their corresponding
        observables (objects, ..) attributes and populates input dataset with.
        Args:
            dataset: dataset object to populate
        """
        if clear_existing_data:
            dataset.clear_data()
        if not self.set_images_lists:
            images = filter( lambda path: path.is_file(), self.annotations_dir.iterdir())
            images = [remove_file_extension(path.stem) for path in images]
        else:
            images = []
            images_to_set = {}
            for set_name, set_images_list in self.set_images_lists.items():
                set_images = [remove_file_extension(image) for \
                    image in read_list_file(set_images_list)]
                images += set_images
                for image in set_images:
                    images_to_set[image] = set_name
        for image in tqdm(images):
            annotation = load_xml_annotation(
                self.annotations_dir / Path(image + self.ANN_EXT))

            # Construct raw observables list
            # (= tuples with observable type + observable tree-like structure)
            raw_observables = []
            if isinstance(annotation['annotation']['object'], list):
                raw_observables += [('object', obj) for obj in annotation['annotation']['object']]
            else:
                raw_observables.append(('object', annotation['annotation']['object']))

            # Construct image attributes
            image_attributes = {'name': image}
            for const in self._entries_constructors:
                if const.process == 'lookup':
                    valid, attr_val = const.extract_in(annotation)
                    if not valid:
                        raise ImpossibleAttributeExtraction(const, annotation)
                    image_attributes[const.name] = attr_val
                else:
                    if const.name == 'filename':
                        image_attributes[const.name] = image + '.' + self.img_ext
                    elif const.name == 'dir':
                        image_attributes[const.name] = self.images_dir
                    elif const.name == 'set':
                        if self.set_images_lists:
                            image_attributes['set'] = images_to_set[image]
                        else:
                            image_attributes['set'] = const.default_val
                    else:
                        raise NotImplementedError

            # Extract attributes for each observable
            observables = []
            for obs_type, obs_data in raw_observables:
                observable = {'type': obs_type}
                for const in self._obs_constructors:
                    if const.observable_type != obs_type:
                        continue
                    if const.process == 'lookup':
                        valid, attr_val = const.extract_in(obs_data)
                        if not valid:
                            raise ImpossibleAttributeExtraction(const, obs_data)
                        observable[const.name] = attr_val
                    else:
                        raise NotImplementedError
                observables.append(observable)

            dataset.add_entry(image_attributes, observables=observables)
