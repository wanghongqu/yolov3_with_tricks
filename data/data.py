import numpy as np
import os
import config as cfg
import random

from data.utils import load_annotations, parse_annotation, resize_to_train_size


class Data:
    def __init__(self, is_training=True):
        self.batch_num = 0
        os.system('mkdir -p ' + cfg.CHECKPOINT_PATH)
        os.system('mkdir -p ' + cfg.LOG_PATH)
        if is_training:
            self.annotations = load_annotations(cfg.DATA_PATH + '/2007_trainval/') + \
                               load_annotations(cfg.DATA_PATH + '/2012_trainval/')
        else:
            self.annotations = load_annotations(cfg.DATA_PATH + '/2007_test/')
        self.annotations = np.array(self.annotations)
        self.total_batch = int(len(self.annotations) / float(cfg.BATCH_SIZE) + 0.5)

    def __next__(self):
        train_input_size = random.choice(cfg.TRAIN_INPUT_SIZE)
        train_output_size = train_input_size / cfg.STRIDES

        batch_image = np.zeros((cfg.BATCH_SIZE, train_input_size, train_input_size, 3), dtype=np.float32)
        batch_label_sbbox = np.zeros(
            (cfg.BATCH_SIZE, train_output_size[0], train_output_size[0], cfg.PRED_NUM_PER_GRID, 6 + 20))
        batch_label_mbbox = np.zeros(
            (cfg.BATCH_SIZE, train_output_size[1], train_output_size[1], cfg.PRED_NUM_PER_GRID, 6 + 20))
        batch_label_lbbox = np.zeros(
            (cfg.BATCH_SIZE, train_output_size[2], train_output_size[2], cfg.PRED_NUM_PER_GRID, 6 + 20))

        batch_annotations = self.annotations[self.batch_num * cfg.BATCH_SIZE:(self.batch_num + 1) * cfg.BATCH_SIZE]
        for line in batch_annotations:
            image, boxes = parse_annotation(line)
            image, boxes = resize_to_train_size(image, boxes, train_input_size)

            # mix_up
            if random.ranom() < 0.5:
                mix_idx = random.randint(0, len(self.annotations) - 1)
                mix_img, mix_boxes = parse_annotation(self.annotations[mix_idx])
                pass

        self.batch_num += 1
        if (self.batch_num >= self.total_batch):
            self.batch_num = 0
            np.random.shuffle(self.annatations)
        return None

        def __iter__(self):
            return self
