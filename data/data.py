import xml.etree.cElementTree as ET
import config as cfg
import os

from data.utils import load_annotations


class Data:
    def __init__(self):
        os.system('mkdir -p' + cfg.CHECKPOINT_PATH)
        train_annatations = load_annotations(cfg.DATA_PATH + '/2007_trainval/') + \
                            load_annotations(cfg.DATA_PATH + '/2012_trainval/')
        test_annatations = load_annotations(cfg.DATA_PATH + '/2007_test/')

    pass


def __next__(self):
    pass


def __iter__(self):
    return self
