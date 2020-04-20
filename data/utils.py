import config as cfg
import xml.etree.cElementTree as ET
import os


def load_annotations(data_path, classes, use_difficult_bbox=False):
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
            class_ind = classes.index(obj.find('name').text.lower().strip())
            xmin = bbox.find('xmin').text.strip()
            xmax = bbox.find('xmax').text.strip()
            ymin = bbox.find('ymin').text.strip()
            ymax = bbox.find('ymax').text.strip()
            annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
        annotations.append(annotation)
    return annotations
