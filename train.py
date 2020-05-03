import os

import numpy as np
import config as cfg
import tensorflow as tf
from data.data import Data
from eval.detect import YoloDetect
from model.loss import yolo_loss
from model.net import get_yolo_model
from model.utils import get_lr
from tensorflow.train import Checkpoint, CheckpointManager
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
model = get_yolo_model()
optimizer = tf.optimizers.Adam()
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
train_data = Data()
test_data = Data(is_training=False)
writer = tf.summary.create_file_writer(cfg.LOG_PATH)
writer.set_as_default()
test_loss = []
start = tf.Variable(initial_value=0)
check_point = Checkpoint(m=model, optim=optimizer, s=start)
manager = CheckpointManager(check_point, cfg.CHECKPOINT_PATH, 1)
detect = YoloDetect(model=model)
if (cfg.RESTORE_TRAINING and tf.train.latest_checkpoint(cfg.CHECKPOINT_PATH)):
    print('loaded the previous checkpoints form CHECKPOINT PATH!')
    check_point.restore(tf.train.latest_checkpoint(cfg.CHECKPOINT_PATH))

if start.numpy() >= 20:
    for layer in model.layers:
        trainable = True
    print('unfrezze all layers to training')

for i in tf.range(start.numpy(), cfg.EPOCHS):
    tf.print('epoch:', i)
    for image, label_sbbox, label_mbbox, label_lbbox in train_data:
        if (i.numpy() < cfg.WARM_UP_EPOCHS):
            optimizer.lr = 1e-5
        else:
            lr = get_lr(optimizer.iterations, train_data.get_size() // cfg.BATCH_SIZE)
            optimizer.lr = lr
        with tf.GradientTape() as tape:
            pred_sbbox, pred_mbbox, pred_lbbox = model(tf.convert_to_tensor(image, dtype=tf.float32))
            loss_val = yolo_loss(pred_sbbox, pred_mbbox, pred_lbbox, label_sbbox, label_mbbox, label_lbbox)
            scaled_loss = optimizer.get_scaled_loss(loss_val)
        scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        grads = optimizer.get_unscaled_gradients(scaled_gradients)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # tf.summary.scalar('train_loss', loss_val, optimizer.iterations)
    if (optimizer.iterations % 50 == 0 and optimizer.iterations):
        tf.print(optimizer.iterations, 'train_loss', loss_val, 'lr:', optimizer.lr)
detect.detect_image(
    r'/content/yolov3_with_tricks/data/VOC/2007_trainval/JPEGImages/000007.jpg 141,50,500,330,6')
for image, label_sbbox, label_mbbox, label_lbbox in test_data:
    pred_sbbox, pred_mbbox, pred_lbbox = model(image)
    loss_val = yolo_loss(pred_sbbox, pred_mbbox, pred_lbbox, label_sbbox, label_mbbox, label_lbbox)
    test_loss.append(loss_val.numpy())
start.assign(tf.constant(i, dtype=tf.int32))
manager.save()
print("test loss:", np.mean(test_loss))
test_loss = []
os.system('zip -r checkpoints' + str(i.numpy()) + '.zip /content/yolov3_with_tricks/logs/checkpoints/')
os.system('mv checkpoints*.zip /content/drive/My\ Drive/Colab\ Notebooks/yolo_tricks/')
# tf.summary.scalar('test_loss', loss_val / test_data.get_size(), optimizer.iterations)
# tf.summary.scalar('lr', optimizer.lr, step=optimizer.iterations)
# test_loss.assign(tf.constant(0, dtype=tf.float32))
