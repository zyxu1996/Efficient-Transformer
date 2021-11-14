import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import os
from tqdm import tqdm

save_path = './heatmap/uperhead/'
if not os.path.exists(save_path):
    os.mkdir(save_path)


def draw_features(x, savename):
    tic = time.time()
    b, c, h, w = x.shape
    for i in tqdm(range(int(c))):
        img = x[0, i, :, :].cpu().numpy()
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  # change value [0, 1] to [0, 255]
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # generate heat map
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        if not os.path.exists(os.path.join(save_path, savename)):
            os.mkdir(os.path.join(save_path, savename))
        cv2.imwrite(os.path.join(save_path, savename, savename + '_' + str(i) + '.png'), img)
    plt.close()
    print("{} time:{}".format(savename, time.time()-tic))
