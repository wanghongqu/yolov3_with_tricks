import config as cfg
import tensorflow as tf
from data.data import Data
from model.loss import yolo_loss
from model.net import get_yolo_model
from model.utils import get_lr

model = get_yolo_model()
optimizer = tf.optimizers.Adam()
train_data = Data()
test_data = Data(is_training=False)
writer = tf.summary.create_file_writer(cfg.LOG_PATH)
writer.set_as_default()
test_loss = tf.Variable(initial_value=0)
for i in range(cfg.EPOCHS):
    for image, label_sbbox, label_mbbox, label_lbbox in train_data:
        lr = get_lr(optimizer.iterations, train_data.get_size() // cfg.BATCH_SIZE)
        optimizer.lr = lr
        with tf.GradientTape() as tape:
            pred_sbbox, pred_mbbox, pred_lbbox = model(tf.convert_to_tensor(image, dtype=tf.float32))
            loss_val = yolo_loss(pred_sbbox, pred_mbbox, pred_lbbox, label_sbbox, label_mbbox, label_lbbox)
        grads = tape.gradient(loss_val, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        tf.summary.scalar('train_loss', loss_val, optimizer.iterations)
    for image, label_sbbox, label_mbbox, label_lbbox in test_data:
        pred_sbbox, pred_mbbox, pred_lbbox = model(image)
        loss_val = yolo_loss(pred_sbbox, pred_mbbox, pred_lbbox, label_sbbox, label_mbbox, label_lbbox)
        tf.summary.scalar('test_loss', loss_val, optimizer.iterations)
