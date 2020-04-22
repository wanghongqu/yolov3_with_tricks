import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
import config as cfg

def separable_conv(input, output_c, strides=1, kernel_size=3):
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same',
                        depthwise_initializer=tf.random_normal_initializer(stddev=0.01))(input)
    x = BatchNormalization(gamma_initializer=tf.ones_initializer, beta_initializer=tf.zeros_initializer)(x)
    x = tf.nn.relu6(x)
    x = conv_bn_relu(x, filters=output_c, kernel_size=1, strides=1, padding='same')
    return x


def conv_bn_relu(x, filters, kernel_size=3, strides=1, padding='same', bn=True, activation=True):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
               kernel_initializer=tf.random_normal_initializer(stddev=0.01))(x)
    if bn:
        x = BatchNormalization(beta_initializer=tf.zeros_initializer, gamma_initializer=tf.ones_initializer)(x)
    if activation:
        x = tf.nn.relu6(x)
    return x


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
    pred_raw_class_prob = pred[5:]
    pred_class_prob = tf.sigmoid(pred_raw_class_prob)

    y = tf.tile(tf.range(grid_size)[:, tf.newaxis], [1, grid_size])[..., tf.newaxis]
    x = tf.tile(tf.range(grid_size)[tf.newaxis, :], [grid_size, 1])[..., tf.newaxis]
    grid = tf.concat([x, y], axis=-1)[tf.newaxis, :, :, tf.newaxis, 2]
    grid = tf.cast(grid, dtype=tf.float32)

    pred_xminymin = strides * (grid + 0.5 - pred_dx1dy1)
    pred_xmaxymax = strides * (grid + 0.5 + pred_dx2dy2)

    ret = tf.zeros_like(pred)
    ret[..., :2] = pred_xminymin
    ret[..., 2:4] = pred_xmaxymax
    ret[..., 4] = pred_conf
    ret[..., 5:] = pred_class_prob
    return ret

# def get_lr(step):
#     if(step<cfg.WARM_UP_PERIOD*)
#     pass

'''
pred = self.__global_step < warmup_steps,
true_fn = lambda: self.__global_step / warmup_steps * self.__learn_rate_init,
false_fn = lambda: self.__learn_rate_end + 0.5 * (self.__learn_rate_init - self.__learn_rate_end) *
(1 + tf.cos((self.__global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))'''
