"""
Generic module to populate a Dataset object from a standard segmentation database.
A standard segmentation database is made of the following elements :
- some images stored at 1 or multiple locations on disk
- some images represented segmentation maps of the images database (usually at png lossless
  compression format with same size than their corresponding image and 1 channel. Value
  of segmentation map at pixel (u, v) is an integer representing the id of the class
  associated with pixel (u, v).
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any

from tqdm import tqdm
from PIL import Image

from datum.datasets import Dataset
from datum.readers import DatasetReader, AttributeConstructor, ObsAttributeConstructor
from datum.readers import update_obs_attr_constructors, update_entries_attr_constructors
from datum.utils.io_utils import read_list_file
from datum.utils.annotations_utils import remove_file_extension
from datum.utils.exceptions import ImpossibleAttributeExtraction


class SegmentationDatasetReader(DatasetReader):
    """Class to populate a Dataset object from a standard Segmentation disk dataset.

    Attributes:
        img_ext: image files extension.
        segmap_ext: segmentation map images files extension.
        sets: dictionary mapping images sets names to tuples with their corresponding
            images directory and segmentation maps directory.
        images_dir: path to directory containing images to load. Is not used if sets is passed.
        segmaps_dir: path to directory containing segmentation maps corresponding
            to images in images_dir. Is not used if sets is passed.
        images_list: file containing list of list of images names to load with their corresponding
            segmentation maps. Is used to filter images from images_dir or sets[<set>]['images_dir'].
        _entries_constructors: list of attributes constructors for each image.
        _obs_constructors: list of observable constructors.
    """

    # Default dataset entries and observables attributes to extract
    DEFAULT_ENTRIES_CONSTRUCTORS = [AttributeConstructor('filename', 'custom', str, None, False),
                                    AttributeConstructor('dir', 'custom', Path, None, False),
                                    AttributeConstructor('width', 'custom', int, None, False,
                                                         ['annotation', 'size', 'width']),
                                    AttributeConstructor('height', 'custom', int, None, False,
                                                         ['annotation', 'size', 'height']),
                                    AttributeConstructor('set', 'custom', str, 'unknown', False),
                                    AttributeConstructor('segmap_path', 'custom', Path, None, False)]
    DEFAULT_OBS_CONSTRUCTORS = []

    def __init__(self,
                 img_ext: str, segmap_ext: str,
                 sets: Optional[Dict[str, Tuple[Path, Path]]] = None,
                 images_dir: Optional[Path] = None, segmaps_dir: Optional[Path] = None,
                 images_list: Optional[Union[List[str], Path]] = None,
                 entries_constructors: Optional[List[AttributeConstructor]] = None,
                 obs_constructors: Optional[List[ObsAttributeConstructor]] = None) -> None:
        if (images_dir and not segmaps_dir) or (segmaps_dir and not images_dir):
            raise ValueError('"images_dir" and "segmap_dir" should be passed together')
        if not images_dir and not sets:
            raise ValueError('At least one of | images_dir + segmaps_dir | sets |'
                             'should be passed.')

        if entries_constructors:
            seg_entries_constructors = \
                update_obs_attr_constructors(self.DEFAULT_ENTRIES_CONSTRUCTORS,
                                                   entries_constructors)
        else:
            seg_entries_constructors = self.DEFAULT_ENTRIES_CONSTRUCTORS

        if obs_constructors:
            seg_obs_constructors = \
                update_obs_attr_constructors(self.DEFAULT_OBS_CONSTRUCTORS, obs_constructors)
        else:
            seg_obs_constructors = self.DEFAULT_OBS_CONSTRUCTORS
        super().__init__(None,
                         entries_constructors=seg_entries_constructors,
                         obs_constructors=seg_obs_constructors)

        self.img_ext: str = img_ext
        self.segmap_ext: str = segmap_ext
        self.images_dir: Path = images_dir
        self.segmaps_dir: Path = segmaps_dir 
        self.sets: Dict[str, Tuple[Path, Path]] = sets
        if images_list:
            if isinstance(images_list, Path):
                self.images_list: List[str] = read_list_file(images_list)
            elif isinstance(images_list, List):
                self.images_list: List[str] = images_list
        else:
            self.images_list = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        d = {'reader_type': type(self).__name__,
             'root_dir': str(self.root_dir),
             'entries_constructors': [const.to_dict() for const in self._entries_constructors],
             'img_ext': self.img_ext,
             'segmap_ext': self.segmap_ext,
             'images_dir': str(self.images_dir),
             'segmaps_dir': str(self.segmaps_dir)}
        if self._obs_constructors:
            d['obs_constructors'] = [const.to_dict() for const in self._obs_constructors]
        if self.sets:
            d['sets'] = {}
            for set_name, (set_images_dir, set_seg_dir) in self.sets.items():
                d['sets'][set_name] = (str(set_images_dir), str(set_seg_dir))
        return d

    def feed(self, dataset: Dataset, clear_existing_data: bool = False) -> None:
        """Extracts all segmentation individual entries (images) and their corresponding
        observables (segmentation map -> path to the segmentation map) and populates input
        dataset with.
        Args:
            dataset: dataset object to populate
        """
        if clear_existing_data:
            dataset.clear_data()
        images_data = {}
        img_ext_regexp = r'.*.' + self.img_ext
        if self.sets:
            for set_name, (set_images_dir, set_segmaps_dir) in self.sets.items():
                images_files = filter(lambda path: path.is_file(), set_images_dir.iterdir())
                images_files = [image_file for image_file in images_files
                                if re.match(img_ext_regexp, str(image_file))]
                for image_file in images_files:
                    image_name = remove_file_extension(image_file.name)
                    images_data[image_name] = {'file': image_file,
                                               'set': set_name,
                                               'segmap_dir': set_segmaps_dir}
        else:
            images_files = filter(lambda path: path.is_file(), self.images_dir.iterdir())
            images_files = [image_file for image_file in images_files
                            if re.match(img_ext_regexp, str(image_file))]

            for image_file in images_files:
                image_name = remove_file_extension(image_file.name)
                images_data[image_name] = {'file': Path(image_file),
                                           'set': 'unknown',
                                           'segmap_dir': self.segmaps_dir}

        # Check that all images in images_list are in images_data
        if self.images_list and set(self.images_list).difference(set(images_data.keys())):
            raise ValueError('Some images from images_list don\'t exist') 

        # Construct images attributes and add images to dataset
        for image, image_data in tqdm(images_data.items()):
            if self.images_list and image not in self.images_list:
                continue
            image_attributes = {'name': image}
            for const in self._entries_constructors:
                if const.process == 'lookup':
                    raise NotImplementedError
                image_pil = Image.open(image_file)
                if const.name == 'filename':
                    image_attributes[const.name] = image_data['file'].name
                elif const.name == 'dir':
                    image_attributes[const.name] = image_data['file'].parent
                elif const.name == 'width':
                    image_attributes[const.name] = image_pil.size[0]
                elif const.name == 'height':
                    image_attributes[const.name] = image_pil.size[1]
                elif const.name == 'set':
                    image_attributes[const.name] = image_data['set']
                elif const.name == 'segmap_path':
                    segmap_path = image_data['segmap_dir'] / (image + '.' + self.segmap_ext)
                    if not segmap_path.exists():
                        if not const.set_to_default:
                            raise ImpossibleAttributeExtraction(const, image_data)
                    else:
                        image_attributes[const.name] = segmap_path
                else:
                    raise NotImplementedError
            dataset.add_entry(image_attributes)
