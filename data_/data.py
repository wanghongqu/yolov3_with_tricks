import xml.etree.cElementTree as ET
import os
import config as cfg
import sys

from data_.utils import load_annotations


class Data:
    def __init__(self, is_training=True):
        os.system('mkdir -p' + cfg.CHECKPOINT_PATH)
        if is_training:
            annatations = load_annotations(cfg.DATA_PATH + '/2007_trainval/') + \
                          load_annotations(cfg.DATA_PATH + '/2012_trainval/')
        else:
            annatations = load_annotations(cfg.DATA_PATH + '/2007_test/')
        print(annatations[0])
        pass

    def __next__(self):
        return None

    def __iter__(self):
        return self
