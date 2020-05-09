import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
import config as cfg
import tensorflow as tf


def separable_conv(input, output_c, strides=1, kernel_size=3):
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same',
                        depthwise_initializer=tf.random_normal_initializer(stddev=0.01))(input)
    x = BatchNormalization(gamma_initializer=tf.ones_initializer, beta_initializer=tf.zeros_initializer)(x)
    x = tf.nn.relu6(x)
    x = conv_bn_relu(x, filters=output_c, kernel_size=1, strides=1, padding='same')
    return x


def conv_bn_relu(x, filters, kernel_size=3, strides=1, padding='same', bn=True, activation=True):
    tmp = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                 kernel_initializer=tf.random_normal_initializer(stddev=0.01))(x)
    if bn:
        tmp = BatchNormalization(beta_initializer=tf.zeros_initializer, gamma_initializer=tf.ones_initializer)(tmp)
    if activation:
        tmp = tf.nn.relu6(tmp)
    return tmp


def conv_bias(input, inp, oup, kernel, stride, padding):
    return Conv2D(inp, oup, kernel, stride, padding, bias=True)(input)


def decode(pred, strides):
    '''
    :param pred: [batch_size,grid_size,grid_size,3*25]
    :return:
    '''
    grid_size = pred.shape[1]
    pred = tf.reshape(pred, shape=[-1, grid_size, grid_size, 3, 25])
    pred_raw_dx1dy1 = pred[..., :2]
    pred_dx1dy1 = tf.exp(pred_raw_dx1dy1)
    pred_raw_dx2dy2 = pred[..., 2:4]
    pred_dx2dy2 = tf.exp(pred_raw_dx2dy2)
    pred_raw_conf = pred[..., 4:5]
    pred_conf = tf.sigmoid(pred_raw_conf)
    pred_raw_class_prob = pred[..., 5:]
    pred_class_prob = tf.sigmoid(pred_raw_class_prob)

    y = tf.tile(tf.range(grid_size)[:, tf.newaxis], [1, grid_size])[..., tf.newaxis]
    x = tf.tile(tf.range(grid_size)[tf.newaxis, :], [grid_size, 1])[..., tf.newaxis]
    grid = tf.concat([x, y], axis=-1)[tf.newaxis, :, :, tf.newaxis, :]
    grid = tf.cast(grid, dtype=tf.float32)

    pred_xminymin = strides * (grid + 0.5 - pred_dx1dy1)
    pred_xmaxymax = strides * (grid + 0.5 + pred_dx2dy2)

    ret = tf.concat([pred_xminymin, pred_xmaxymax, pred_conf, pred_class_prob], axis=-1)
    return ret


def multi_step_decay(step, step_per_epoch):
    warmup_steps = cfg.WARM_UP_EPOCHS * step_per_epoch
    if (step < warmup_steps):
        return step / warmup_steps * cfg.LEARN_RATE_INIT
    elif step >= step_per_epoch * cfg.MILESTONES[0] and step <= step_per_epoch * cfg.MILESTONES[1]:
        return cfg.LEARN_RATE_INIT * 0.1
    elif step > step_per_epoch * cfg.MILESTONES[1]:
        return cfg.LEARN_RATE_INIT * 0.001
    else:
        return cfg.LEARN_RATE_INIT


def get_lr(step, step_per_epoch):
    warmup_steps = cfg.WARM_UP_EPOCHS * step_per_epoch
    train_steps = cfg.EPOCHS * step_per_epoch
    if (step < warmup_steps):
        return step / warmup_steps * cfg.LEARN_RATE_INIT
    return cfg.LEARN_RATE_END + 0.5 * (cfg.LEARN_RATE_INIT - cfg.LEARN_RATE_END) * (
            1 + np.math.cos((step - warmup_steps) / (train_steps - warmup_steps) * np.pi))


def cal_diou(pred, label):
    # 计算 pred面积
    pred_wh = pred[..., 2:4] - pred[..., :2]
    pred_area = pred_wh[..., 0:1] * pred_wh[..., 1:2]  # batch_size,grid,grid,3,1
    # 计算 label面积
    label_wh = label[..., 2:4] - label[..., 0:2]  # batch_size,grid,grid,3,2
    label_area = label_wh[..., 0:1] * label_wh[..., 1:2]  # batch_size,grid,grid,3,1
    # 计算交集
    intersect_minxy = tf.maximum(pred[..., :2], label[..., :2])
    intersect_maxxy = tf.minimum(pred[..., 2:4], label[..., 2:4])
    intersect_wh = tf.maximum(intersect_maxxy - intersect_minxy, 0.0)
    intersect_area = intersect_wh[..., 0:1] * intersect_wh[..., 1:2]
    iou = intersect_area / (label_area + pred_area - intersect_area)  # batch_size,grid,grid,3,1

    outer_minxy = tf.minimum(pred[..., :2], label[..., :2])
    outer_maxxy = tf.maximum(pred[..., 2:4], label[..., 2:4])
    outer_wh = tf.maximum(outer_maxxy - outer_minxy, 0.0)
    outer_eye_dis = tf.sqrt(tf.pow(outer_wh[..., 0:1], 2) + tf.pow(outer_wh[..., 1:2], 2))

    label_center = (label[..., 2:4] + label[..., 0:2]) / 2
    pred_center = (pred[..., 2:4] + pred[..., 0:2]) / 2
    inter_wh = pred_center - label_center
    inter_eye_dis = tf.sqrt(tf.pow(inter_wh[..., 0:1], 2) + tf.pow(inter_wh[..., 1:2], 2))

    return iou - inter_eye_dis / outer_eye_dis


def cal_giou(pred, label):
    # 计算 pred面积
    pred_wh = pred[..., 2:4] - pred[..., :2]
    pred_area = pred_wh[..., 0:1] * pred_wh[..., 1:2]  # batch_size,grid,grid,3,1
    # 计算 label面积
    label_wh = label[..., 2:4] - label[..., 0:2]  # batch_size,grid,grid,3,2
    label_area = label_wh[..., 0:1] * label_wh[..., 1:2]  # batch_size,grid,grid,3,1
    # 计算交集
    intersect_minxy = tf.maximum(pred[..., :2], label[..., :2])
    intersect_maxxy = tf.minimum(pred[..., 2:4], label[..., 2:4])
    intersect_wh = tf.maximum(intersect_maxxy - intersect_minxy, 0.0)
    intersect_area = intersect_wh[..., 0:1] * intersect_wh[..., 1:2]
    iou = intersect_area / (label_area + pred_area - intersect_area)  # batch_size,grid,grid,3,1
    # 外部闭集
    outer_minxy = tf.minimum(pred[..., :2], label[..., :2])
    outer_maxxy = tf.maximum(pred[..., 2:4], label[..., 2:4])
    outer_wh = tf.maximum(outer_maxxy - outer_minxy, 0.0)
    out_area = outer_wh[..., 0:1] * outer_wh[..., 1:2]

    giou = 1.0 * (iou - (out_area - label_area - pred_area + intersect_area) / out_area)  # grid,grid,3,1
    return giou


def calc_iou(pred, object_boxes):
    expand_pred = tf.expand_dims(pred, axis=-2)  # batch_size,grid,grid,3,1,4
    # pred area
    pred_wh = expand_pred[..., 2:4] - expand_pred[..., :2]
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]  # shape:[grid,grid,3,1]

    # object area
    obj_wh = object_boxes[..., 2:4] - object_boxes[..., :2]
    obj_area = obj_wh[..., 0] * obj_wh[..., 1]  # shape:[N]

    # intersect area

    intersect_minxy = tf.maximum(expand_pred[..., :2], object_boxes[..., :2])  # shape:grid,grid,3,N,2
    intersect_maxxy = tf.minimum(expand_pred[..., 2:4], object_boxes[..., 2:4])  # shape:grid,grid,3,N,2
    intersect_wh = tf.maximum(intersect_maxxy - intersect_minxy, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    return intersect_area / (obj_area + pred_area - intersect_area)
