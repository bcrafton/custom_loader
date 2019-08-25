
import matplotlib.pyplot as plt
import numpy as np
from draw_boxes import draw_boxes

image = np.random.uniform(size=(1, 448, 448, 3))
predict = np.random.uniform(size=(1,7,7,10))

draw_boxes('test.jpg', image, predict)
