import numpy as np
import os
import config as cfg
import random
import tensorflow as tf
from data.utils import load_annotations, parse_annotation, resize_to_train_size


class Data:
    def __init__(self, is_training=True):
        self.batch_num = 0
        if not os.path.exists(cfg.CHECKPOINT_PATH):
            os.system('mkdir -p ' + cfg.CHECKPOINT_PATH)
        if not os.path.exists(cfg.LOG_PATH):
            os.system('mkdir -p ' + cfg.LOG_PATH)
        if is_training:
            self.annotations = load_annotations(cfg.DATA_PATH + '2007_trainval/') + \
                               load_annotations(cfg.DATA_PATH + '2012_trainval/')
        else:
            self.annotations = load_annotations(cfg.DATA_PATH + '2007_test/', is_training=False)
        self.annotations = np.array(self.annotations)
        self.total_batch = int(len(self.annotations) / float(cfg.BATCH_SIZE) + 0.5)
        self.is_training = is_training
        print('total  data: ', len(self.annotations))
        print('batch_size:', cfg.BATCH_SIZE)
        if is_training:
            print('total training epoch:', cfg.EPOCHS)

    def __next__(self):
        if self.is_training:
            train_input_size = random.choice(cfg.TRAIN_INPUT_SIZE)
            train_output_size = [train_input_size // v for v in cfg.STRIDES]
        else:
            train_input_size = cfg.TEST_INPUT_SIZE
            train_output_size = [train_input_size // v for v in cfg.STRIDES]

        batch_image = np.zeros((cfg.BATCH_SIZE, train_input_size, train_input_size, 3), dtype=np.float32)
        batch_label_sbbox = np.zeros(
            (cfg.BATCH_SIZE, train_output_size[0], train_output_size[0], cfg.PRED_NUM_PER_GRID, 6 + 20))
        batch_label_mbbox = np.zeros(
            (cfg.BATCH_SIZE, train_output_size[1], train_output_size[1], cfg.PRED_NUM_PER_GRID, 6 + 20))
        batch_label_lbbox = np.zeros(
            (cfg.BATCH_SIZE, train_output_size[2], train_output_size[2], cfg.PRED_NUM_PER_GRID, 6 + 20))

        batch_annotations = self.annotations[self.batch_num * cfg.BATCH_SIZE:(self.batch_num + 1) * cfg.BATCH_SIZE]

        for i, line in enumerate(batch_annotations):
            image, boxes = parse_annotation(line, self.is_training)
            image, boxes = resize_to_train_size(image, boxes, train_input_size)

            # mix_up
            if random.random() < 0.5 and self.is_training:
                mix_idx = random.randint(0, len(self.annotations) - 1)
                mix_img, mix_boxes = parse_annotation(self.annotations[mix_idx])
                mix_img, mix_boxes = resize_to_train_size(mix_img, mix_boxes, train_input_size)
                lam = np.random.beta(1.5, 1.5)
                image = lam * image + (1 - lam) * mix_img
                boxes = np.concatenate([boxes, lam * np.ones((len(boxes), 1), dtype=np.float32)], axis=-1)
                mix_boxes = np.concatenate([mix_boxes, (1 - lam) * np.ones((len(mix_boxes), 1), dtype=np.float32)],
                                           axis=-1)
                boxes = np.concatenate([boxes, mix_boxes], axis=0)
            else:
                boxes = np.concatenate([boxes, np.ones((len(boxes), 1), dtype=np.float32)], axis=-1)
            s_label, m_label, l_label = self.creat_label(boxes, train_output_size)
            batch_image[i, :, :, :] = image.astype(np.float32)/255.0
            batch_label_sbbox[i, :, :, :, :] = s_label
            batch_label_mbbox[i, :, :, :, :] = m_label
            batch_label_lbbox[i, :, :, :, :] = l_label
        self.batch_num += 1
        if (self.batch_num >= self.total_batch):
            self.batch_num = 0
            np.random.shuffle(self.annotations)
            raise StopIteration()
        return tf.convert_to_tensor(batch_image, dtype=tf.float32), tf.convert_to_tensor(batch_label_sbbox,dtype=tf.float32), tf.convert_to_tensor(
            batch_label_mbbox, dtype=tf.float32), tf.convert_to_tensor(batch_label_lbbox, dtype=tf.float32)

    def __iter__(self):
        return self

    def get_size(self):
        return len(self.annotations)

    def creat_label(self, boxes, train_output_size):
        label = [np.zeros(dtype=np.float32,
                          shape=(train_output_size[i], train_output_size[i], cfg.PRED_NUM_PER_GRID, 6 + 20)) for i
                 in range(3)]
        ground_truth_count = [np.zeros(shape=(train_output_size[i], train_output_size[i])) for i in range(3)]
        for i in range(3):
            label[i][..., -1] = 1.0
        for box in boxes:
            xy = box[..., :4]
            cls = box[..., 4]
            w = box[..., -1]

            wh = xy[2:] - xy[:2]
            area = np.sqrt(wh[0] * wh[1])
            if (area < 30):
                branch = 0
            elif (area < 90 and area >= 30):
                branch = 1
            else:
                branch = 2

            one_hot = np.zeros(dtype=np.float32, shape=(20,))
            one_hot[int(cls)] = 1
            one_hot = one_hot * (1 - cfg.DELTA) + (1 - one_hot) * cfg.DELTA / 20
            x_, y_ = (xy[..., :2] + xy[..., 2:4]) / 2
            x_ = int(x_ / cfg.STRIDES[branch])
            y_ = int(y_ / cfg.STRIDES[branch])
            grid_total = ground_truth_count[branch][y_, x_]
            if (grid_total >= 2.8):
                continue;
            ground_truth_count[branch][y_, x_] += 1
            label[branch][y_, x_, int(grid_total), :] = np.concatenate([xy, [1], one_hot, [w]], axis=-1)
        s_label, m_label, l_label = label
        return s_label, m_label, l_label
