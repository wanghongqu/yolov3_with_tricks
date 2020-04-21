import random
import numpy as np
import config as cfg
import numpy as np
import xml.etree.cElementTree as ET
from PIL import Image, ImageEnhance
import os


def load_annotations(data_path, use_difficult_bbox=False):
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', 'trainval.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    annotations = []
    for image_ind in image_inds:
        image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg')
        annotation = image_path
        label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
        root = ET.parse(label_path).getroot()
        objects = root.findall('object')
        for obj in objects:
            difficult = obj.find('difficult').text.strip()
            if (not use_difficult_bbox) and (int(difficult) == 1):
                continue
            bbox = obj.find('bndbox')
            class_ind = cfg.CLASSES.index(obj.find('name').text.lower().strip())
            xmin = bbox.find('xmin').text.strip()
            xmax = bbox.find('xmax').text.strip()
            ymin = bbox.find('ymin').text.strip()
            ymax = bbox.find('ymax').text.strip()
            annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
        annotations.append(annotation)
    return annotations


def parse_annotation(line):
    path = line.split()[0]
    boxes = np.array([[float(v) for v in part.split(',')] for part in line.split()[1:]])
    image = np.array(Image.open(path))
    image = random_left_right_flip(image, boxes)
    return image, boxes


def random_left_right_flip(image, boxes):
    if random.random() < 0.5:
        image[...] = image[:, ::-1, :]
        ih, iw = image.shape[0], image.shape[1]
        boxes[:, [0, 2]] = boxes[:, [2, 0]]
        boxes[:, [0, 2]] = iw - boxes[:, [0, 2]]
    return image, boxes


def random_crop(image, boxes):
    dxdy_left = np.min(boxes[..., :2], axis=-2)
    dxdy_right = np.max(boxes[..., 2:4], axis=-2)

    random_dx_left = np.random.uniform(0, dxdy_left[0])
    random_dy_left = np.random.uniform(0, dxdy_left[1])
    random_dx_right = np.random.uniform(image.shape[1], dxdy_right[0])
    random_dy_right = np.random.uniform(image.shape[1], dxdy_right[1])

    image = image[:random_dx_right, :random_dy_right, :]
    image = image[random_dx_left:, random_dy_left:, :]
    return image

def random_shift():
    pass


#
# image = Image.open(r'C:\Users\LenovoPC\PycharmProjects\eat_tensorflow_in_20_days\aa.jpg')
# print(image.size)
# print(np.array(image).shape)
