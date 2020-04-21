import os

# yolo
TRAIN_INPUT_SIZE = [320 + i * 32 for i in range(10)]
PRED_NUM_PER_GRID = 3

# train
BATCH_SIZE = 1024
EPOCHS = 300
STRIDES = [8, 16, 32]

# test
TEST_INPUT_SIZE = 544

# name and path
CHECKPOINT_PATH = './logs/checkpoints/'
LOG_PATH = './logs/logs/'
PROJECT_PATH = os.getcwd()
DATA_PATH = os.path.join(os.getcwd(), 'data/VOC/')
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
