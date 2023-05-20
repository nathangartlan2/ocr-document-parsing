from dataclasses import dataclass
import tensorflow as tf
import keras


@dataclass
class ImageInput:
    """Class keeps track of an image and its filename for labeling"""
    fileName: str
    image: object


def __init__(self, fileName: str, image: object):
    self.name = fileName
    self.image = fileName
