import numpy as np
import tensorflow as tf
from PIL import Image

import config as cfg
from data.utils import resize_to_train_size, draw_image_with_boxes
from eval.utils import iou_calc1, nms
from model.net import get_yolo_model
from model.utils import decode


class YoloDetect:
    def __init__(self, model=None):
        if model == None:
            self.model = get_yolo_model()
            checkpoint = tf.train.Checkpoint(m=self.model)
            # checkpoint.restore(tf.train.latest_checkpoint(r"C:\Users\LenovoPC\Desktop\checkpoints"))
            checkpoint.restore(tf.train.latest_checkpoint(cfg.CHECKPOINT_PATH))
        else:
            self.model = model

    def detect_image(self, annotation_line):
        tf.keras.backend.set_learning_phase(0)
        '''
        :param data:[1,cfg.TEST_INPUT_SIZE,cfg.TEST_INPUT_SIZE,3]
        :return:
        '''
        image = Image.open(annotation_line.split()[0])
        image = np.array(image)
        resized_image = resize_to_train_size(image.copy(), cfg.TEST_INPUT_SIZE, is_training=False) / 255.0
        pred_sbbox, pred_mbbox, pred_lbbox = self.model((resized_image[np.newaxis, :, :, :]).astype(np.float32))
        pred_sbbox = tf.reshape(decode(pred_sbbox, 8), shape=[-1, 25])
        pred_mbbox = tf.reshape(decode(pred_mbbox, 16), shape=[-1, 25])
        pred_lbbox = tf.reshape(decode(pred_lbbox, 32), shape=[-1, 25])
        pred_boxes_info = tf.concat([pred_sbbox, pred_mbbox, pred_lbbox], axis=0)

        pred_class_prob = pred_boxes_info[..., 4:5] * pred_boxes_info[..., 5:]
        assert pred_class_prob.shape == (pred_boxes_info.shape[0], 20)
        pred_msk = tf.reduce_max(pred_boxes_info[..., 4:5] * pred_boxes_info[..., 5:], axis=-1) > cfg.IGNORE_THRESH
        # assert pred_msk.shape == (pred_boxes_info.shape[0], 1)
        pred_class_prob = pred_class_prob[pred_msk]
        print("after masked:", pred_class_prob.shape[0])
        pred_boxes_coor = (pred_boxes_info[..., :4][pred_msk]).numpy()
        if not (pred_boxes_coor.shape[0]):
            return
        pred_msk = np.logical_and(pred_boxes_coor[:, 0] < pred_boxes_coor[:, 2],
                                  pred_boxes_coor[:, 1] < pred_boxes_coor[:, 3])
        pred_class_prob = (pred_class_prob[pred_msk]).numpy()
        pred_boxes_coor = pred_boxes_coor[pred_msk]
        pred_boxes_coor[:, 0:2] = np.maximum([0.0, 0.0], pred_boxes_coor[:, 0:2])
        pred_boxes_coor[:, 2:4] = np.minimum([cfg.TEST_INPUT_SIZE - 1, cfg.TEST_INPUT_SIZE - 1],
                                             pred_boxes_coor[:, 2:4])
        if (pred_boxes_coor.shape[0]):
            print(pred_boxes_coor.shape[0])
        clazz = np.argmax(pred_class_prob, axis=-1)
        prob = pred_class_prob[np.arange(len(pred_class_prob)), clazz]

        pred = np.concatenate([pred_boxes_coor, prob[:, np.newaxis], clazz[:, np.newaxis]], axis=-1)

        draw_image_with_boxes(resize_to_train_size(image.copy(), cfg.TEST_INPUT_SIZE, is_training=False), nms(pred),
                              'test.png')
        tf.keras.backend.set_learning_phase(1)
