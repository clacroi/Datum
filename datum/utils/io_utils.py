"""
Datum I/O utility functions.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from lxml import etree
from xml.etree import ElementTree
import xml.dom.minidom

from datum.utils.annotations_utils import etree_to_dict, dict_to_etree


def load_xml_annotation(file: Path, remove_encoding: bool = True) -> dict:
    """Reads a txt file representing a XML structure and loads it in a tree-like structure.
    Each XML node with a unique name will be represented by an equivalent node
    in the resulting structure, that is to say a (name, sub_tree) pair where name is
    the XML node name and sub_tree the tree-like substructure from the node in the XML file.
    XML nodes with same name will be represented by a List of (name, List[sub_tree]) where
    name is the common name and List[sub_tree] a list of tree-like substructures from the
    multiple nodes having the same name.
    Args:
        file: Path to XML annotaation file.
        remove_encoding: Whether to remove encoding header in XML file string.
    Returns:
        Dict representing the XML file tree structure.
    """
    if file.exists():
        with open(file) as fd:
            ann_str = fd.read()
        if remove_encoding:
            ann_str = ann_str.replace(" encoding='utf8'", "")
        try:
            return etree_to_dict(etree.fromstring(ann_str))
        except ValueError:
            print("Error while parsing annotation file, returning None")
            return None
    else:
        raise Exception('XML annotation file {} not found'.format(file))


def load_json_kibana_data(file: Path) -> Dict[str, Any]:
    """
    Read data from a Kibana Elastic-Search request single-entry JSON file and load it into a dict.
    If available, data from Rekognition about images and objects tags are pre-formatted
    into a common form.

    Args:
        file: Path to Kibana Elastic-Search request single-entry JSON file.
    Returns:
        Dict representing the data from the results file in a tree structure.
    """
    if not file.exists():
        raise Exception('Json Kibana annotation file {} does not exist'.format(file))
    with open(file, 'r') as f:
        data = json.load(f)

    # Construct 'object' and 'label' sections in result with pre-extracted data in a
    # "pre-formatted form" : they will be used to construct 'object' and 'label' observables
    data['object'] = []
    data['label'] = []
    if 'labels' not in data:
        return data

    for label in data['labels']:
        # TODO : also add any kind of (key, value) pairs found in Kibana Json "labels" section
        reformatted_label = {'name': label['Name'],
                             'score': label['Confidence'] / 100.0}
        data['label'].append(reformatted_label)

        if label['Instances']:
            for instance in label['Instances']:
                reformtatted_obj = {'name': label['Name'],
                                    'score': instance['Confidence'] / 100.0,
                                    'xmin': instance['BoundingBox']['Left'],
                                    'ymin': instance['BoundingBox']['Top'],
                                    'xmax': instance['BoundingBox']['Left'] + instance['BoundingBox']['Width'],
                                    'ymax': instance['BoundingBox']['Top'] + instance['BoundingBox']['Height']}
                data['object'].append(reformtatted_obj)

    return data


def prettify(tree: ElementTree) -> str:
    """
    Return a pretty-printed XML string from input ElementTree, i.e a string
    with correct tabs for better visualizing XMLs.
    Requires tools from xml.dom.minidom package
    Args:
        tree: input ElementTree.
    Returns:
        Pretty string with correct indentations for writing input tree.
    """
    raw_string = ElementTree.tostring(tree, 'utf-8')
    reparsed = xml.dom.minidom.parseString(raw_string)

    return reparsed.toprettyxml(indent="\t")


def dump_xml_annotation(annotation: dict, out_file: Path) -> None:
    """Writes to out_file a XML string representing input tree-like structure annotation
    Args:
        annotation: dict representing XML tree-like structure to write.
        out_file: Path to write resulting XML.
    """
    ann_etree = dict_to_etree(annotation)

    with open(out_file, 'w', encoding="utf-8") as out:
        out.write(prettify(ann_etree))


def read_list_file(file: Path) -> None:
    """Reads txt file with multiple rows and constructs a list from it.
    Args:
        file: Path to input txt file.
    """
    if file.exists():
        with open(file, 'r+') as f:
            lines = f.readlines()
        return [line.rstrip("\n") for line in lines]
    else:
        raise Exception('List file {} not found'.format(file))

def dump_list_to_file(list_to_dump: List, file: Path) -> None:
    """Write list_to_dump in a txt file with 1 row per element in input list.
    Args:
        list_to_dump: list of elements to dump.
        file: Path to input txt file.
    """
    with open(file, 'w') as f:
        f.write("\n".join(list_to_dump))
