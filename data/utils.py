import random
import numpy as np
import config as cfg
import numpy as np
import xml.etree.cElementTree as ET
from PIL import Image, ImageEnhance
import os
from PIL import ImageDraw


def load_annotations(data_path, use_difficult_bbox=False, is_training=True):
    if is_training:
        img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', 'trainval.txt')
    else:
        img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', 'test.txt')
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


def parse_annotation(line, is_training=True):
    path = line.split()[0]
    boxes = np.array([[float(v) for v in part.split(',')] for part in line.split()[1:]])
    image = np.array(Image.open(path))
    if is_training:
        image, boxes = random_left_right_flip(image, boxes)
        image, boxes = random_crop(image, boxes)
        image, boxes = random_shift(image, boxes)
    return image, boxes


def random_left_right_flip(image, boxes):
    if random.random() < 0.5:
        image[...] = image[:, ::-1, :]
        ih, iw = image.shape[0], image.shape[1]
        boxes[:, [0, 2]] = boxes[:, [2, 0]]
        boxes[:, [0, 2]] = iw - 1 - boxes[:, [0, 2]]
    return image, boxes


def random_crop(image, boxes):
    assert (np.max(boxes[:, 2]) <= image.shape[1])
    assert (np.max(boxes[:, 3]) <= image.shape[0])
    xy_min = np.min(boxes[:, :2], axis=-2)
    xy_max = np.max(boxes[:, 2:4], axis=-2)

    random_dx_min = int(np.random.uniform(0, xy_min[0]))
    random_dy_min = int(np.random.uniform(0, xy_min[1]))
    random_dx_max = int(np.random.uniform(xy_max[0], image.shape[1] - 1))
    random_dy_max = int(np.random.uniform(xy_max[1], image.shape[0] - 1))

    image = image[random_dy_min:random_dy_max + 1, random_dx_min:random_dx_max + 1, :]
    boxes[:, [0, 2]] = boxes[:, [0, 2]] - random_dx_min
    boxes[:, [1, 3]] = boxes[:, [1, 3]] - random_dy_min
    # print("box_maxxy:",boxes[:,2:4])
    # print("xy_min,xy_max",xy_min,xy_max)
    # print("random_data:",random_dx_min,random_dy_min,random_dx_max,random_dy_max)

    return image, boxes


def random_shift(image, boxes):
    ih, iw = image.shape[:2]
    dxdy_min = np.min(boxes[..., :2], axis=-2)

    dxdy_max = image.shape[:2][::-1] - np.max(boxes[..., 2:4], axis=-2) - 1
    new_image = Image.new('RGB', (iw, ih), color=(128, 128, 128))
    random_dx = int(np.random.uniform(-dxdy_min[0], dxdy_max[0]))
    random_dy = int(np.random.uniform(-dxdy_min[1], dxdy_max[1]))
    boxes[:, [0, 2]] = boxes[:, [0, 2]] + random_dx
    boxes[:, [1, 3]] = boxes[:, [1, 3]] + random_dy
    if (random_dx < 0):
        image = image[:, -random_dx:, :]
        random_dx = 0
    if (random_dx > 0):
        image = image[:, :-random_dx]
    if (random_dy < 0):
        image = image[-random_dy:, :, :]
        random_dy = 0
    if (random_dy > 0):
        image = image[:-random_dy, :, :]
    new_image.paste(Image.fromarray(image), (random_dx, random_dy))

    return np.array(new_image), boxes


def resize_to_train_size(image, train_input_size, boxes=None, is_training=True):
    ih, iw = image.shape[:2]
    scale = min(train_input_size / ih, train_input_size / iw)
    new_h, new_w = int(scale * ih), int(scale * iw)
    image = Image.fromarray(image)
    image = image.resize((new_w, new_h))
    new_image = Image.new('RGB', [train_input_size, train_input_size], (128, 128, 128))
    dx, dy = int((train_input_size - new_w) / 2.0), int((train_input_size - new_h) / 2.0)
    new_image.paste(image, (dx, dy))
    if (is_training):
        boxes[..., [0, 2]] = scale * boxes[..., [0, 2]] + dx
        boxes[..., [1, 3]] = scale * boxes[..., [1, 3]] + dy
        return np.array(new_image), boxes
    else:
        return np.array(new_image)


def draw_image_with_boxes(image, boxes, name):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box[:4].astype(np.int32).tolist(), width=2, outline='yellow')
    image.save(name)
