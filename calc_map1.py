
import numpy as np
import matplotlib.pyplot as plt

##############################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

offset = [
[[0, 0], [0, 64], [0, 128], [0, 192], [0, 256], [0, 320], [0, 384]], 
[[64, 0], [64, 64], [64, 128], [64, 192], [64, 256], [64, 320], [64, 384]], 
[[128, 0], [128, 64], [128, 128], [128, 192], [128, 256], [128, 320], [128, 384]], 
[[192, 0], [192, 64], [192, 128], [192, 192], [192, 256], [192, 320], [192, 384]], 
[[256, 0], [256, 64], [256, 128], [256, 192], [256, 256], [256, 320], [256, 384]],  
[[320, 0], [320, 64], [320, 128], [320, 192], [320, 256], [320, 320], [320, 384]],  
[[384, 0], [384, 64], [384, 128], [384, 192], [384, 256], [384, 320], [384, 384]]
]

def calc_iou(label, pred1, pred2):
    iou1 = calc_iou_help(label, pred1)
    iou2 = calc_iou_help(label, pred2)
    return np.stack([iou1, iou2], 3)

def calc_iou_help(boxA, boxB):
    intersectionX = np.minimum(boxA[:, :, :, 0] + boxA[:, :, :, 2], boxB[:, :, :, 0] + boxB[:, :, :, 2]) - np.maximum(boxA[:, :, :, 0], boxB[:, :, :, 0])
    intersectionY = np.minimum(boxA[:, :, :, 1] + boxA[:, :, :, 3], boxB[:, :, :, 1] + boxB[:, :, :, 3]) - np.maximum(boxA[:, :, :, 1], boxB[:, :, :, 1])
    intersection = np.maximum(0., intersectionX) * np.maximum(0., intersectionY)
    union = (boxA[:, :, :, 2] * boxA[:, :, :, 3]) + (boxB[:, :, :, 2] * boxB[:, :, :, 3]) - intersection
    iou = intersection / union
    return iou

def grid_to_pix(box):
    box[:, :, :, 0:2] = 64.  * box[:, :, :, 0:2] + offset
    box[:, :, :, 2:4] = 448. * box[:, :, :, 2:4]
    return box

'''
def mAP(label, pred, conf_thresh=0.5, iou_thresh=0.5):

    label = np.reshape(grid_to_pix(label[:, :, :, 0:5]), (-1, 7, 7, 5))
    pred1 = np.reshape(grid_to_pix(pred[:, :, :, 0:5]),   (1, 7, 7, 5))
    pred2 = np.reshape(grid_to_pix(pred[:, :, :, 5:10]),  (1, 7, 7, 5))

    iou = calc_iou(label, pred1, pred2)
    resp_box = iou[:, :, :, 0] > iou[:, :, :, 1]

    obj = label[:, :, :, 4]
    pred_conf1 = pred1[:, :, :, 4]
    pred_conf2 = pred2[:, :, :, 4]

    ###############################

    iou_mask = obj * np.max(iou, axis=3) > iou_thresh
    conf_mask = np.where(resp_box, np.ones_like(obj) * pred_conf1, np.ones_like(obj) * pred_conf2) > conf_thresh

    TP = np.count_nonzero(iou_mask * conf_mask)
    TP_FP = np.count_nonzero(pred_conf1 > conf_thresh) + np.count_nonzero(pred_conf2 > conf_thresh)
    TP_FN = np.count_nonzero(obj)

    return TP, TP_FP, TP_FN
'''

def mAP(label, pred, conf_thresh=0.5, iou_thresh=0.5):

    label = np.reshape(grid_to_pix(label[:, :, :, 0:5]), (-1, 7, 7, 5))
    pred1 = np.reshape(grid_to_pix(pred[:, :, :, 0:5]),   (1, 7, 7, 5))
    pred2 = np.reshape(grid_to_pix(pred[:, :, :, 5:10]),  (1, 7, 7, 5))

    conf = np.stack((pred1[:, :, :, 4], pred2[:, :, :, 4]), axis=3)
    conf = np.reshape(conf, (7, 7, 2))

    # TODO: this is not correct bc if box1 already used, we can use box2 even if it has smaller iou ... I think ...
    obj = label[:, :, :, 4]
    iou = calc_iou(label, pred1, pred2)
    # (box1 iou > box2 iou) & (box1 iou > iou_thresh) & obj
    correct1 = (iou[:, :, :, 0] > iou[:, :, :, 1]) * (iou[:, :, :, 0] > iou_thresh) * obj
    # (box2 iou > box1 iou) & (box2 iou > iou_thresh) & obj
    correct2 = (iou[:, :, :, 1] > iou[:, :, :, 0]) * (iou[:, :, :, 1] > iou_thresh) * obj
    correct = np.max(np.stack((correct1, correct2), axis=3), axis=0)

    return conf, correct, objs

##############################################################

results = np.load('results.npy', allow_pickle=True).item()

##############################################################

confs = []
corrects = []

total_correct = 0

for batch in range(100):
    pred_name = 'pred%d' % (batch)
    pred = results[pred_name]

    label_name = 'label%d' % (batch)
    label = results[label_name]
    coords, objs, no_objs, cats, vlds = label

    #############

    for ex in range(8):
        l = coords[ex] * np.expand_dims(vlds[ex], axis=3)
        p = np.reshape(pred[ex], (1, 7, 7, 90))
        conf, correct, obj = mAP(l, p, conf_thresh=-1e6, iou_thresh=0.3)
        confs.append(conf); corrects.append(correct)

    total_correct += np.count_nonzero(objs)

##############################################################

confs = np.reshape(confs, -1)
corrects = np.reshape(corrects, -1)

argsort = np.argsort(confs)[::-1]
confs = confs[argsort]
corrects = corrects[argsort]

##############################################################

precision = np.cumsum(corrects) / np.array(range(1, len(corrects)+1))
recall = np.cumsum(corrects) / total_correct

plt.plot(recall, precision)
plt.show()



















