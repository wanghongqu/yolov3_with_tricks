import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model

from model.utils import conv_bn_relu, separable_conv, decode


def get_backbone():
    base_model = keras.applications.mobilenet_v2.MobileNetV2(alpha=1.0, include_top=False,
                                                             weights='imagenet', input_shape=(416, 416, 3),
                                                             input_tensor=None, pooling=None, classes=1000)
    return keras.Model(base_model.input, [
        base_model.get_layer('block_6_expand').input,
        base_model.get_layer('block_11_expand').input,
        base_model.get_layer('out_relu').output,
    ])


def get_yolo_model():
    backbone = get_backbone()
    for layer in backbone.layers:
        layer.trainable = False
    featuremap_small, featuremap_medium, featuremap_large = backbone.output

    conv = conv_bn_relu(featuremap_large, 512, kernel_size=1)
    conv = separable_conv(conv, output_c=1024)
    conv = conv_bn_relu(conv, 512, kernel_size=1)
    conv = separable_conv(conv, output_c=1024)
    conv = conv_bn_relu(conv, 512, kernel_size=1)

    # detection branch of large objects
    conv_lbbox = separable_conv(conv, 1024)
    pred_large_box = conv_bn_relu(conv_lbbox, 3 * 25, kernel_size=1, bn=False, activation=False)

    # upsample and merge
    conv = conv_bn_relu(conv, 256, kernel_size=1)
    conv = UpSampling2D()(conv)
    conv = Concatenate()([conv, featuremap_medium])

    conv = conv_bn_relu(conv, filters=256, kernel_size=1)
    conv = separable_conv(conv, 512)
    conv = conv_bn_relu(conv, 256, kernel_size=1)
    conv = separable_conv(conv, 512)
    conv = conv_bn_relu(conv, 256, kernel_size=1)

    # detection branch of medium objects
    conv_mbbox = separable_conv(conv, 512)
    pred_mbbox = conv_bn_relu(conv_mbbox, 3 * 25, kernel_size=1, bn=False, activation=False)

    # upsample and merge
    conv = conv_bn_relu(conv, 128, kernel_size=1)
    conv = UpSampling2D()(conv)
    conv = Concatenate()([featuremap_small, conv])

    conv = conv_bn_relu(conv, 128, kernel_size=1)
    conv = separable_conv(conv, 256)
    conv = conv_bn_relu(conv, 128, kernel_size=1)
    conv = separable_conv(conv, 256)
    conv = conv_bn_relu(conv, 128, kernel_size=1)

    # detection branch of samll objects
    conv_sbbox = separable_conv(conv, 256)
    pred_sbbox = conv_bn_relu(conv_sbbox, 3 * 25, kernel_size=1, activation=False, bn=False)

    return keras.Model(backbone.input, [pred_sbbox, pred_mbbox, pred_large_box])

#
# backbone = get_backbone()
# for layer in backbone.layers:
#     print(layer.output_shape,'           ',layer.name)
# featuremap_small, featuremap_medium, featuremap_large = backbone.output
# print(featuremap_small.shape)
# print(featuremap_medium.shape)
# print(featuremap_large.shape)
