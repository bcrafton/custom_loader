
import matplotlib.pyplot as plt
import numpy as np

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

def draw_boxes(name, image, predict, det, iou):
    # check shapes
    assert(np.shape(image) == (1, 448, 448, 3))
    image = np.reshape(image, (448, 448, 3))
    assert(np.shape(predict) == (1, 7, 7, 90))
    predict = np.reshape(predict, (7, 7, 90))

    coords, objs, no_objs, cats = det

    image = image / np.max(image)
    top = image
    bottom = np.copy(image)

    pred_box1 = predict[:, :, 0:5]
    pred_box2 = predict[:, :, 5:10]

    nbox = min(len(coords), len(colors))
    for ii in range(nbox):
        coord = coords[ii]
        obj = objs[ii]

        ##############################################

        [xc, yc] = np.squeeze(np.where(obj > 0))

        ##############################################

        [x, y, w, h, _] = coord[xc][yc]

        x = int(x * 64. + xc * 64.)
        y = int(y * 64. + yc * 64.)
        w = int(w * 448.)
        h = int(h * 448.)

        [x11, x12, x21, x22] = [x, x+5, x+w-5, x+w]
        [y11, y12, y21, y22] = [y, y+5, y+h-5, y+h]
        top[x11:x12, y12:y21, :] = colors[ii]
        top[x21:x22, y12:y21, :] = colors[ii]
        top[x12:x21, y11:y12, :] = colors[ii]
        top[x12:x21, y21:y22, :] = colors[ii]

        ##############################################

        iou1 = iou[ii][xc][yc][0]
        iou2 = iou[ii][xc][yc][1]

        if iou1 < iou2:
            [x, y, w, h, _] = pred_box2[xc][yc]
            print (name, 'iou: ', iou2, color_names[ii])
        else:
            [x, y, w, h, _] = pred_box1[xc][yc]
            print (name, 'iou: ', iou1, color_names[ii])

        x = int(x * 64. + xc * 64.)
        y = int(y * 64. + yc * 64.)
        w = int(w * 448.)
        h = int(h * 448.)

        [x11, x12, x21, x22] = [x, x+5, x+w-5, x+w]
        [y11, y12, y21, y22] = [y, y+5, y+h-5, y+h]
        bottom[x11:x12, y12:y21, :] = colors[ii]
        bottom[x21:x22, y12:y21, :] = colors[ii]
        bottom[x12:x21, y11:y12, :] = colors[ii]
        bottom[x12:x21, y21:y22, :] = colors[ii]

        ##############################################

    concat = np.concatenate((top, bottom), axis=1)
    plt.imsave(name, concat)







