"""
Common annotations files utility functions.
"""

from typing import List, Tuple, Dict, Any
from collections import defaultdict
from xml.etree import cElementTree as ET
from xml.etree import ElementTree


VIDEO_EXTENSIONS = ["mp4", "MP4", "webm", "WEBM", "mkv", "MKV",
                    "avi", "AVI", "mpeg", "MPEG", "mpg", "MPG"]
IMAGE_EXTENSIONS = ["jpeg", "JPEG", "jpg", "JPG", "png", "PNG", "tiff", "TIFF", "tif", "TIF", "pgm", "PGM"]
EXTENSIONS = VIDEO_EXTENSIONS + IMAGE_EXTENSIONS


def remove_file_extension(filename: str,
                          extensions: List = EXTENSIONS) -> str:
    """
    Remove end of filenames extensions from input filename (".something").
    Args:
        filename: input filename.
        extensions: list of extensions to look for and remove.
    Returns:
        string without file extension
    """
    out = filename
    for extension in extensions:
        if "." + extension in out:
            idx = out.find("." + extension)
            out = out[0:idx]
            break
    return out


def etree_to_dict(t: ElementTree) -> Dict:
    """Converts etree XML structure to Python dictionary
    Args:
        t: input etree to convert
    Returns:
    :   dictionary constructed from input etree
    """
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag] = text
        else:
            d[t.tag] = text
    return d


def dict_to_etree(d: Dict) -> ElementTree:
    """Converts Python dictionary to etree XML structure. Input dict
    Args:
        d: input dict to convert.
    Returns:
    :   dictionary constructed from input etree
    """
    def _to_etree(d, root):
        if isinstance(d, dict):
            for k, v in d.items():
                assert isinstance(k, str)
                # (i) v is a list : multiple SubElement must be constructed
                if isinstance(v, list):
                    for elt in v:
                        _to_etree(elt, ET.SubElement(root, k))
                # (ii) v is not a list
                else:
                    _to_etree(v, ET.SubElement(root, k))
        else:
            root.text = str(d)

    assert isinstance(d, dict) and len(d) == 1

    tag, body = next(iter(d.items()))
    root = ET.Element(tag)
    _to_etree(body, root)

    return root

def get_dict_value(in_dict: Dict, path: List):
    """Extracts a value in input dictionary at given path if path is correct
    Args:
        in_dict: dict to look for the value in.
        path: sequence of keys to get to value in in_dict.
    Returns:
        Value at given path if correct or None if path incorrect
    """
    d = in_dict
    for tag in path:
        if tag not in d.keys():
            return None
        d = d[tag]
    return d

def set_dict_value(in_dict: Dict, path: List, val: Any) -> Any:
    """Sets a value in input dictionary at given path
    Args:
        in_dict: dict to set value in.
        path: sequence of keys to set to value in in_dict.
    Returns:
    """
    if not path:
        return val

    key = path[0]
    if key in in_dict:
        set_dict_value(in_dict[key], path[1:], val)
    else:
        in_dict[key] = set_dict_value({}, path[1:], val)
    return in_dict
