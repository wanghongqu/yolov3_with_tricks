import os

# yolo
TRAIN_INPUT_SIZE = [320 + i * 32 for i in range(10)]
PRED_NUM_PER_GRID = 3
DELTA = 0.01

# train
BATCH_SIZE = 16
EPOCHS = 200
STRIDES = [8, 16, 32]
WARM_UP_EPOCHS = 2
LEARN_RATE_INIT = 2e-4
LEARN_RATE_END = 1e-6
IOU_LOSS_THRESH = 0.5

RESTORE_TRAINING = True

# test
TEST_INPUT_SIZE = 544
IGNORE_THRESH = 0.02
IOU_THRESH = 0.45

# name and path
CHECKPOINT_PATH = './logs/checkpoints/'
LOG_PATH = './logs/logs/'
PROJECT_PATH = os.getcwd()
DATA_PATH = os.path.join(os.getcwd(), 'data/VOC/')
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']