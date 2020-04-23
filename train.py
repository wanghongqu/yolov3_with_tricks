import config as cfg
import tensorflow as tf
from data.data import Data
from model.loss import yolo_loss
from model.net import get_yolo_model
from model.utils import get_lr
from tensorflow.train import Checkpoint, CheckpointManager

model = get_yolo_model()
optimizer = tf.optimizers.Adam()
train_data = Data()
test_data = Data(is_training=False)
writer = tf.summary.create_file_writer(cfg.LOG_PATH)
writer.set_as_default()
test_loss = tf.Variable(initial_value=0, dtype=tf.float32)
start = tf.Variable(initial_value=0)
check_point = Checkpoint(m=model, optim=optimizer, s=start)
manager = CheckpointManager(check_point, cfg.CHECKPOINT_PATH, 3)

if (cfg.RESTORE_TRAINING):
    check_point.restore(tf.train.latest_checkpoint(cfg.CHECKPOINT_PATH))

for i in tf.range(start.numpy(), cfg.EPOCHS):
    tf.print('epoch:', i)
    for image, label_sbbox, label_mbbox, label_lbbox in train_data:
        lr = get_lr(optimizer.iterations, train_data.get_size() // cfg.BATCH_SIZE)
        optimizer.lr = lr
        with tf.GradientTape() as tape:
            pred_sbbox, pred_mbbox, pred_lbbox = model(tf.convert_to_tensor(image, dtype=tf.float32))
            loss_val = yolo_loss(pred_sbbox, pred_mbbox, pred_lbbox, label_sbbox, label_mbbox, label_lbbox)
        grads = tape.gradient(loss_val, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # tf.summary.scalar('train_loss', loss_val, optimizer.iterations)
        if(optimizer.iterations%100==0):
            tf.print(optimizer.iterations, 'train_loss', loss_val, 'lr:', lr)
    for image, label_sbbox, label_mbbox, label_lbbox in test_data:
        pred_sbbox, pred_mbbox, pred_lbbox = model(image)
        loss_val = yolo_loss(pred_sbbox, pred_mbbox, pred_lbbox, label_sbbox, label_mbbox, label_lbbox)
        test_loss.assign_add(loss_val)
    tf.print(optimizer.iterations, "test loss:", loss_val / (test_data.get_size() / cfg.BATCH_SIZE))
    # tf.summary.scalar('test_loss', loss_val / test_data.get_size(), optimizer.iterations)
    # tf.summary.scalar('lr', optimizer.lr, step=optimizer.iterations)
    # test_loss.assign(tf.constant(0, dtype=tf.float32))
    manager.save()
