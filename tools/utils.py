import numpy as np


def label_to_RGB(image):
    RGB = np.zeros(shape=[image.shape[0], image.shape[1], 3], dtype=np.uint8)
    index = image == 0
    RGB[index] = np.array([255, 255, 255])
    index = image == 1
    RGB[index] = np.array([0, 0, 255])
    index = image == 2
    RGB[index] = np.array([0, 255, 255])
    index = image == 3
    RGB[index] = np.array([0, 255, 0])
    index = image == 4
    RGB[index] = np.array([255, 255, 0])
    index = image == 5
    RGB[index] = np.array([255, 0, 0])
    return RGB