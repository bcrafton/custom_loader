
import csv
import numpy as np

label = '''1,1,912,484,97,109,0,7,1
2,1,912,484,97,109,0,7,1
3,1,912,484,97,109,0,7,1
4,1,912,484,97,109,0,7,1
5,1,912,484,97,109,0,7,1
6,1,912,484,97,109,0,7,1
7,1,912,484,97,109,0,7,1
8,1,912,484,97,109,0,7,1
9,1,912,484,97,109,0,7,1'''

for line in label.splitlines():
    print (line)

'''
with open('/home/brian/Documents/projects/object_detection/MOT17Det/train/MOT17-02/gt/gt.txt', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
'''

mat = np.loadtxt(open('/home/brian/Documents/projects/object_detection/MOT17Det/train/MOT17-02/gt/gt.txt', "rb"), delimiter=",", skiprows=0)
print (mat)
