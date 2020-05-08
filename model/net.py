import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model

from model.utils import conv_bn_relu, separable_conv, decode, conv_bias


def get_backbone():
    base_model = keras.applications.mobilenet_v2.MobileNetV2(alpha=1.0, include_top=False,
                                                             weights='imagenet', input_tensor=None, pooling=None,
                                                             classes=1000)
    return keras.Model(base_model.input, [
        base_model.get_layer('block_6_expand').input,
        base_model.get_layer('block_11_expand').input,
        base_model.get_layer('out_relu').output,
    ])


'''
l0=torch.ones(1,512,10,10)
l1=torch.ones(1,256,20,20)
l2=torch.ones(1,128,40,40)
'''


def asff(level, x_level_0, x_level_1, x_level_2, rfb=False, vis=False):
    dim = [512, 256, 128]
    inter_dim = dim[level]
    if level == 0:
        x_level_0_resized = x_level_0
        x_level_1_resized = conv_bn_relu(x_level_1, inter_dim, kernel_size=3, strides=2, padding='same')
        x_level_2_downsampled = MaxPooling2D(pool_size=3, padding='same', strides=2)(x_level_2)
        x_level_2_resized = conv_bn_relu(x_level_2_downsampled, inter_dim, kernel_size=3, strides=2, padding='same')
    elif level == 1:
        x_level_0_compressed = conv_bn_relu(x_level_0, inter_dim, 1, 1)
        x_level_0_resized = UpSampling2D(size=(2, 2))(x_level_0_compressed)
        x_level_1_resized = x_level_1
        x_level_2_resized = conv_bn_relu(x_level_2, inter_dim, 3, 2)
    elif level == 2:
        x_level_0_expand = conv_bn_relu(x_level_0, inter_dim, kernel_size=1, strides=1)
        x_level_0_resized = UpSampling2D(size=4)(x_level_0_expand)
        x_level_1_compressed = conv_bn_relu(x_level_1, inter_dim, kernel_size=1)
        x_level_1_resized = UpSampling2D((2, 2))(x_level_1_compressed)
        x_level_2_resized = x_level_2
        pass
    compress_c = 8 if rfb else 16
    level_0_weight_v = conv_bn_relu(x_level_0_resized, compress_c, 1)
    level_1_weight_v = conv_bn_relu(x_level_1_resized, compress_c, 1)
    level_2_weight_v = conv_bn_relu(x_level_2_resized, compress_c, 1)

    levels_weight_v = Concatenate()([level_0_weight_v, level_1_weight_v, level_2_weight_v])
    levels_weight_v = conv_bn_relu(levels_weight_v, 3)
    levels_weight_v = tf.nn.softmax(levels_weight_v, dim=-1)

    fused_out_reduced = x_level_0_resized * levels_weight_v[..., 0:1] + \
                        x_level_1_resized * levels_weight_v[..., 1:2] + \
                        x_level_2_resized * levels_weight_v[..., 2:3]
    out = conv_bn_relu(fused_out_reduced, dim[level], 3)
    return out


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
    conv1 = conv

    # upsample and merge
    conv = conv_bn_relu(conv, 256, kernel_size=1)
    conv = UpSampling2D()(conv)
    conv = Concatenate()([conv, featuremap_medium])

    conv = conv_bn_relu(conv, filters=256, kernel_size=1)
    conv = separable_conv(conv, 512)
    conv = conv_bn_relu(conv, 256, kernel_size=1)
    conv = separable_conv(conv, 512)
    conv = conv_bn_relu(conv, 256, kernel_size=1)
    conv2 = conv

    # upsample and merge
    conv = conv_bn_relu(conv, 128, kernel_size=1)
    conv = UpSampling2D()(conv)
    conv = Concatenate()([featuremap_small, conv])

    conv = conv_bn_relu(conv, 128, kernel_size=1)
    conv = separable_conv(conv, 256)
    conv = conv_bn_relu(conv, 128, kernel_size=1)
    conv = separable_conv(conv, 256)
    conv = conv_bn_relu(conv, 128, kernel_size=1)
    conv3 = conv

    # detection branch of large objects
    conv1 = asff(0, conv1, conv2, conv3)
    conv_lbbox = separable_conv(conv1, 1024)
    pred_large_box = conv_bn_relu(conv_lbbox, 3 * 25, kernel_size=1, bn=False, activation=False)

    # detection branch of medium objects
    conv2 = asff(1, conv1, conv2, conv3)
    conv_mbbox = separable_conv(conv2, 512)
    pred_mbbox = conv_bn_relu(conv_mbbox, 3 * 25, kernel_size=1, bn=False, activation=False)

    # detection branch of samll objects
    conv3 = asff(2, conv1, conv2, conv3)
    conv_sbbox = separable_conv(conv3, 256)
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
