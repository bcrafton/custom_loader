
from LoadCOCO import LoadCOCO
import matplotlib.pyplot as plt
import numpy as np

loader = LoadCOCO()

while True:
    if not loader.empty():
        images, (coords, objs, no_objs, cats) = loader.pop()
        print (np.shape(image))
        print (np.shape(coords))
