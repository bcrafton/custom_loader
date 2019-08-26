
import matplotlib.pyplot as plt
import numpy as np

def draw_boxes(name, image, predict):
    # check shapes
    assert(np.shape(image) == (1, 448, 448, 3))
    image = np.reshape(image, (448, 448, 3))
    assert(np.shape(predict) == (1, 7, 7, 10))
    # find 5 largest confidence values
    predict = np.reshape(predict, [7*7*2,5])
    conf = predict[:,4]
    top1 = np.argsort(conf)[-1:]
    idx = np.unravel_index(top1, shape=[7,7,2])

    [xc, yc, _] = np.squeeze(idx)
    [x, y, w, h, c] = np.squeeze(predict[top1])

    x = int(x * 64. + xc * 64.)
    y = int(y * 64. + yc * 64.)
    w = int(w * 448.)
    h = int(h * 448.)

    image = image / np.max(image)
    top = image
    bottom = np.copy(image)
    bottom[x:(x+w), y:(y+h), :] = 0.0
    concat = np.concatenate((top, bottom), axis=1)
    plt.imsave(name, concat)

