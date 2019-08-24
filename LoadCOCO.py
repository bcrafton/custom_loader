
import os

import csv
import numpy as np

import cv2

import queue
import threading

import json

#########################################

exxact = 1
local = 0

if exxact:
    path = '/home/bcrafton3/Data_HDD/mscoco/'
elif local:
    assert(False)
else:
    assert(False)

#########################################

def get_images(path):

    images = []
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
                for file in folder_files:
                    if 'jpg' in file:
                        full_path = os.path.join(folder_subdir, file)
                        if (full_path not in images):
                            images.append(full_path)

    return images

#########################################

def get_labels_table(json_filename):
    table = {}

    json_file = json.load(json_filename)
    annotations = list(json_file['annotations'])
    for annotation in annotations:
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        image_filename = 'COCO_train2014_%012d.jpg' % (int(image_id))

        if image_filename in table.keys():
            table[image_filename].append(bbox)
        else:
            table[image_filename] = [bbox]

    return table

#########################################

def get_boxes(labels):
    nlabels = len(labels)

    coords  = np.zeros(shape=[nlabels, 16, 9, 5])
    obj     = np.zeros(shape=[nlabels, 16, 9])
    no_obj  = np.zeros(shape=[nlabels, 16, 9])

    for ii in range(nlabels):
        coords[ii, x, y, :] = np.array([l, t, h, w, 1.])
        obj[ii, x, y] = 1.
        no_obj[ii] = np.ones(shape=[16, 9]) - obj[ii]

    return (coords, obj, no_obj)

#########################################

def fill_queue(images, labels_table, q):
    ii = 0
    last = len(images) - 1

    while(True):
        if not q.full():
            filename = images[ii]
            ii = (ii + 1) if (ii < last) else 0

            x = cv2.imread(filename)
            shape = np.shape(x)
            if   shape == (1920, 1080, 3):
                pass
            elif shape == (1080, 1920, 3):
                x = np.transpose(x, [1, 0, 2])
            else:
                print ('skipping: %s' % (filename))
                continue

            x = np.reshape(x, (1, 1920, 1080, 3))

            if filename in labels_table.keys():
                y = get_boxes(labels_table[filename])
            else:
                print ('no label: %s' % (filename))
                continue

            q.put((x, y))

#########################################

class LoadCOCO:

    def __init__(self):
        self.train_images = sorted(get_images(path + 'train_images'))
        # self.test_images = sorted(get_images(path + 'test'))

        self.train_labels_table = get_labels_table(path + 'train_labels/instances_train2014.json')
        # self.test_labels_table = get_labels_table(test_folders)

        self.q = queue.Queue(maxsize=128)
        thread = threading.Thread(target=fill_queue, args=(self.train_images, self.train_labels_table, self.q))
        thread.start()

    def pop(self):
        return self.q.get()

    def empty(self):
        return self.q.empty()

    def full(self):
        return self.q.full()

###################################################################

















