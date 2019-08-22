
import csv
import numpy as np

mat = np.loadtxt(open('/home/brian/Documents/projects/object_detection/MOT17Det/train/MOT17-02/gt/gt.txt', "rb"), delimiter=",", skiprows=0)

lookup = {}
for label in mat:
    key = '/home/brian/Documents/projects/object_detection/MOT17Det/train/MOT17-02/img1/%06d.jpg' % (int(label[0]))
    if key in lookup:
        lookup[key].append(label)
    else:
        lookup[key] = [label]

# print ('%06d' % (100))
# 000100

print (lookup['/home/brian/Documents/projects/object_detection/MOT17Det/train/MOT17-02/img1/000010.jpg'])
print (len(lookup.keys()))
