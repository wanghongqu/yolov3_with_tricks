import os

# yolo
TRAIN_INPUT_SIZE = [320 + i * 32 for i in range(10)]

# train
BATCH_SIZE = 1024
EPOCHS = 300

# test
TEST_INPUT_SIZE = 544

# name and path
PROJECT_PATH = os.getcwd()
DATA_PATH = os.path.join(os.getcwd(), 'data')
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
