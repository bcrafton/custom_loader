
import os
import numpy as np
import cv2

import queue
import threading

# im = cv2.imread("abc.tiff",mode='RGB')

def train_filenames(path):

    training_images = []

    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
                for file in folder_files:
                    training_images.append(os.path.join(folder_subdir, file))

    return training_images

##########################

train_names = train_filenames('/home/brian/Documents/projects/object_detection/MOT17Det/train')
test_names = train_filenames('/home/brian/Documents/projects/object_detection/MOT17Det/test')
print (train_names)

##########################

def fill_queue(d, q):
    ii = 0
    last = len(d) - 1

    while(True):
        if q.full() == False:
            filename = d[ii]
            x = cv2.imread(filename)
            q.put(x)
            ii = (ii + 1) if (ii < last) else 0
            print (ii)

##########################

q = queue.Queue(maxsize=1000)
thread = threading.Thread(target=fill_queue, args=(train_names, q))
thread.start()

##########################

















