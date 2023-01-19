import matplotlib.pyplot as plt

import numpy as np
import cv2 as cv
import os

from utils import get_ax, cluster, resize, getpath, smoothImage, removeGreyBackground
from barsclutering import getContours
import utils as u

def drawBars(im, bars):
    img = im.copy()
    for bar in bars:
        ((x1,y1),(x2,y2)) = bar
        # print((x1,y1),(x2,y2))     
        img = cv.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)
    return img

def getBarsSizes(im, bars):
    img = im.copy()
    im_bars = []
    for (i,bar) in enumerate(bars):
        ((x1,y1),(x2,y2)) = bar
        bar_img = img[y1:y2,x1:x2]
        bar_img = smoothImage(bar_img)
        # img = removeGreyBackground(img)
        bar_img = cluster(bar_img, K=2)
        b, contours, c_mm = getContours(bar_img)
        ((b_x1,b_y1),(b_x2,b_y2)) = c_mm[0]
        # print(len(contours), len(c_mm), "height", b_y2 - b_y1)
        im_bars.append(bar_img)
        # img = cv.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)
    return im_bars

def showImages(imgs):
    size = len(imgs)
    axes = get_ax(2, int(size/2))
    for i in range(0, size):
        axes[int(i/(size/2)), int(i%(size/2))].imshow(imgs[i])
    plt.show() 

if __name__ == '__main__':
    bars_np = []
    with open('bars_original.npy', 'rb') as f:
        bars_np = np.load(f)

    # barrasThiago/azulAltura75.jpg
    # barrasThiago/vermelhoAltura0.jpg
    p = "barrasThiago/azulAltura75.jpg" # getpath()
    img = u.formatSize(cv.imread(p))
    im = img.copy()
    # im = drawBars(im, bars_np)

    im_bars = getBarsSizes(im,bars_np)
    showImages(im_bars)
