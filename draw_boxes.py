
import matplotlib.pyplot as plt
import numpy as np

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

    for ii in range(len(coords)):
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
        red = np.array([1.0, 0.0, 0.0])
        top[x11:x12, y12:y21, :] = red
        top[x21:x22, y12:y21, :] = red
        top[x12:x21, y11:y12, :] = red
        top[x12:x21, y21:y22, :] = red

        ##############################################

        iou1 = iou[ii][xc][yc][0]
        iou2 = iou[ii][xc][yc][1]
        print (name, 'iou1: ', iou1, 'iou2: ', iou2)
        if iou1 < iou2:
            [x, y, w, h, _] = pred_box2[xc][yc]
        else:
            [x, y, w, h, _] = pred_box1[xc][yc]

        x = int(x * 64. + xc * 64.)
        y = int(y * 64. + yc * 64.)
        w = int(w * 448.)
        h = int(h * 448.)

        [x11, x12, x21, x22] = [x, x+5, x+w-5, x+w]
        [y11, y12, y21, y22] = [y, y+5, y+h-5, y+h]
        red = np.array([1.0, 0.0, 0.0])
        bottom[x11:x12, y12:y21, :] = red
        bottom[x21:x22, y12:y21, :] = red
        bottom[x12:x21, y11:y12, :] = red
        bottom[x12:x21, y21:y22, :] = red

        ##############################################

    concat = np.concatenate((top, bottom), axis=1)
    plt.imsave(name, concat)







