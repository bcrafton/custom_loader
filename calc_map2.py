
import numpy as np
import matplotlib.pyplot as plt

##############################################################

colors = [
np.array([1.0, 0.0, 0.0]),
np.array([0.0, 1.0, 0.0]),
np.array([0.0, 0.0, 1.0]),

np.array([1.0, 1.0, 0.0]),
np.array([1.0, 0.0, 1.0]),
np.array([0.0, 1.0, 1.0])
]

color_names = [
'red',
'green',
'blue',
'yellow',
'violet',
'cyan'
]

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

##############################################################

def calc_iou(label, pred1, pred2):
    iou1 = calc_iou_help(label, pred1)
    iou2 = calc_iou_help(label, pred2)
    return np.stack([iou1, iou2], 3)

def calc_iou_help(boxA, boxB):
    intersectionX = np.minimum(boxA[..., 0] + boxA[..., 2], boxB[..., 0] + boxB[..., 2]) - np.maximum(boxA[..., 0], boxB[..., 0])
    intersectionY = np.minimum(boxA[..., 1] + boxA[..., 3], boxB[..., 1] + boxB[..., 3]) - np.maximum(boxA[..., 1], boxB[..., 1])
    intersection = np.maximum(0., intersectionX) * np.maximum(0., intersectionY)
    union = (boxA[..., 2] * boxA[..., 3]) + (boxB[..., 2] * boxB[..., 3]) - intersection
    iou = intersection / union
    return iou

def draw_box(name, image, label, pred, nbox):

    objs      = label[..., 4]
    true_boxs = grid_to_pix(label[..., 0:4])
    boxs1     = grid_to_pix(pred[..., 0:4])
    boxs2     = grid_to_pix(pred[..., 5:9])
    iou = calc_iou(true_boxs, boxs1, boxs2)

    true_image = np.copy(image)
    pred_image = np.copy(image)

    nbox = min(nbox, len(colors))
    for b in range(nbox):
        obj = objs[b]
        [xc, yc] = np.squeeze(np.where(obj > 0))

        box = np.array(true_boxs[b][xc][yc], dtype=int)
        draw_box_help(true_image, box, colors[b])

        iou1 = iou[b][xc][yc][0]
        iou2 = iou[b][xc][yc][1]
        if iou1 > iou2:
            box = np.array(boxs1[xc][yc], dtype=int)
        else:
            box = np.array(boxs2[xc][yc], dtype=int)
        draw_box_help(pred_image, box, colors[b])

    concat = np.concatenate((true_image, pred_image), axis=1)
    plt.imsave(name, concat)

def draw_box_help(image, box, color):
    [x, y, w, h] = box
    [x11, x12, x21, x22] = [x, x+5, x+w-5, x+w]
    [y11, y12, y21, y22] = [y, y+5, y+h-5, y+h]
    image[y11:y12, x12:x21, :] = color
    image[y21:y22, x12:x21, :] = color
    image[y12:y21, x11:x12, :] = color
    image[y12:y21, x21:x22, :] = color

##############################################################

results_filename = 'yolo_coco1_results.npy'
results = np.load(results_filename, allow_pickle=True).item()

##############################################################

for batch in range(5, 100):
    img_name = 'img%d' % (batch)
    imgs = results[img_name]

    pred_name = 'pred%d' % (batch)
    preds = results[pred_name]

    label_name = 'label%d' % (batch)
    labels = results[label_name]
    coords, objs, no_objs, cats, vlds = labels

    #############

    for ex in range(8):
        image = imgs[ex]; image = image / np.max(image)
        label = coords[ex] * np.expand_dims(vlds[ex], axis=3)
        pred = preds[ex]
        nbox = np.count_nonzero(np.average(vlds[ex], axis=(1,2)))
        draw_box('img%d.jpg' % (batch * 8 + ex), image, label, pred, nbox)
        

##############################################################












