import config as cfg
from model.utils import decode, cal_giou, calc_iou
import tensorflow as tf


def yolo_loss(pred_sbbox, pred_mbbox, pred_lbbox, label_sbbox, label_mbbox, label_lbbox):
    loss_val = 0
    loss_val += loss_per_scale(pred_sbbox, label_sbbox, 8)
    loss_val += loss_per_scale(pred_mbbox, label_mbbox, 16)
    loss_val += loss_per_scale(pred_lbbox, label_lbbox, 32)
    return loss_val


def focal(target, actual, alpha=1, gamma=2):
    focal = alpha * tf.pow(tf.abs(target - actual), gamma)
    return focal


def loss_per_scale(pred_raw, label, strides):
    pred = decode(pred_raw, strides)
    pred_raw = tf.reshape(pred_raw, shape=pred.shape)

    input_size = pred.shape[1] * strides
    grid_size = pred.shape[1]
    # 计算xy loss
    giou = cal_giou(pred[..., :4], label[..., :4])
    object_mask = label[..., 4:5]
    wh = (label[..., 2:4] - label[..., :2])
    scale = 2.0 - 1.0 * wh[..., 0:1] * wh[..., 1:2] / float(input_size) / float(input_size)  # grid,grid,3,1
    iou_loss = scale * object_mask * (1.0 - giou)  # batch_size,grid,grid,3,1
    assert iou_loss.shape==(pred.shape[0],grid_size,grid_size,3,1)

    # 计算confidence loss
    ignore_msk = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)
    for i in range(pred.shape[0]):
        object_boxes = tf.boolean_mask(label[i][..., :4], tf.cast(label[i][..., 4], dtype=tf.bool))
        if (object_mask.shape[0] == 0):
            object_boxes = tf.zeros(shape=[1, 4], dtype=tf.float32)
        iou = calc_iou(pred[i][..., :4], object_boxes)  # grid,grid,3,N
        assert iou.shape == (grid_size, grid_size, 3, len(object_boxes))
        max_iou = tf.reduce_max(iou, axis=-1, keepdims=True)  # grid,grid,3,1

        ignore_msk = ignore_msk.write(i, tf.cast(max_iou < cfg.IOU_LOSS_THRESH, dtype=tf.float32))
    ignore_msk = ignore_msk.stack()
    # conf_focal = focal(object_mask, pred[..., 4:5])

    conf_loss = (
            object_mask * tf.nn.sigmoid_cross_entropy_with_logits(label[..., 4:5], pred_raw[..., 4:5]) + (
            1.0 - object_mask) * ignore_msk * tf.nn.sigmoid_cross_entropy_with_logits(label[..., 4:5],
                                                                                      pred_raw[..., 4:5]))
    # batch_size,grid,grid,3,1

    # class prob loss
    class_prob_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(label[..., 5:-1], pred_raw[..., 5:])
    # grid,grid,20
    loss = tf.concat([iou_loss, conf_loss, class_prob_loss], axis=-1) * label[..., -1:]
    # tf.print("iou_loss:", tf.reduce_sum(iou_loss), "conf_loss:", tf.reduce_sum(conf_loss), "class_prob_loss:",
    #  tf.reduce_sum(class_prob_loss))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3, 4]))
    return loss
