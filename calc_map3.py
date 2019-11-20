
import numpy as np

##############################################################

offset = [
[[0, 0], [0, 64], [0, 128], [0, 192], [0, 256], [0, 320], [0, 384]], 
[[64, 0], [64, 64], [64, 128], [64, 192], [64, 256], [64, 320], [64, 384]], 
[[128, 0], [128, 64], [128, 128], [128, 192], [128, 256], [128, 320], [128, 384]], 
[[192, 0], [192, 64], [192, 128], [192, 192], [192, 256], [192, 320], [192, 384]], 
[[256, 0], [256, 64], [256, 128], [256, 192], [256, 256], [256, 320], [256, 384]],  
[[320, 0], [320, 64], [320, 128], [320, 192], [320, 256], [320, 320], [320, 384]],  
[[384, 0], [384, 64], [384, 128], [384, 192], [384, 256], [384, 320], [384, 384]]
]

def grid_to_pix(box):
    box[..., 0:2] = 64.  * box[..., 0:2] + offset
    box[..., 2:4] = 448. * box[..., 2:4]
    return box

def calc_iou(boxA, boxB):
    intersectionX = np.minimum(boxA[..., 0] + boxA[..., 2], boxB[..., 0] + boxB[..., 2]) - np.maximum(boxA[..., 0], boxB[..., 0])
    intersectionY = np.minimum(boxA[..., 1] + boxA[..., 3], boxB[..., 1] + boxB[..., 3]) - np.maximum(boxA[..., 1], boxB[..., 1])
    intersection = np.maximum(0., intersectionX) * np.maximum(0., intersectionY)
    union = (boxA[..., 2] * boxA[..., 3]) + (boxB[..., 2] * boxB[..., 3]) - intersection
    iou = intersection / union
    return iou

##############################################################

def mAP(label, pred, iou_thresh=0.5):

    label = grid_to_pix(label[..., 0:5])
    obj = label[..., 4]
    label = label[np.where(obj == 1)]

    pred1 = grid_to_pix(pred[..., 0:5]).reshape((-1, 5))
    pred2 = grid_to_pix(pred[..., 5:10]).reshape((-1, 5))
    pred = np.concatenate([pred1, pred2], 0)

    ####################

    label = np.reshape(label, (-1, 1, 5))
    pred = np.reshape(pred, (1, -1, 5))
    iou = calc_iou(label, pred)
    correct = np.count_nonzero(np.max(iou > iou_thresh, axis=1))

    ####################
    
    TP = correct
    TP_FP = 7 * 7 * 2
    TP_FN = np.count_nonzero(obj)

    ####################

    return TP, TP_FP, TP_FN

##############################################################

results_filename = 'yolo_coco1.npy'
results = np.load(results_filename, allow_pickle=True).item()

##############################################################

TPs = 0
TP_FPs = 0
TP_FNs = 0

for batch in range(100):
    pred_name = 'pred%d' % (batch)
    pred = results[pred_name]

    label_name = 'label%d' % (batch)
    label = results[label_name]
    coords, objs, no_objs, cats, vlds = label

    #############

    for ex in range(8):
        l = coords[ex] * np.expand_dims(vlds[ex], axis=3) # apply the vld mask
        p = np.reshape(pred[ex], (1, 7, 7, 90))
        TP, TP_FP, TP_FN = mAP(l, p, iou_thresh=0.5)
        TPs += TP; TP_FPs += TP_FP; TP_FNs += TP_FN

##############################################################

precision = TPs / (TP_FPs + 1e-3)
recall = TPs / (TP_FNs + 1e-3)

print ('true positive: %f' % (TPs))
print ('false positive: %f' % (TP_FPs - TPs))
print ('false negative: %f' % (TP_FNs - TPs))
print ()
print ('precision: %f' % (precision))
print ('recall: %f' % (recall))

##############################################################












