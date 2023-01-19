import matplotlib.pyplot as plt
import os
import cv2 as cv
import numpy as np

bars_lim = {"y": [50, 350], "x":[110, 700]}
bars_lim2 = {"y": [0, 260]}

def smoothImage(img, k=19):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (k,k))
    opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    return opening # cv.cvtColor(opening, cv.COLOR_BGR2RGB)

def removeGreyBackground(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    # threshold saturation image
    # thresh1 = cv.threshold(s, 92, 255, cv.THRESH_BINARY)[1]
    thresh1 = cv.threshold(s, 100, 255, cv.THRESH_BINARY)[1]

    # threshold value image and invert
    # thresh2 = cv.threshold(v, 128, 255, cv.THRESH_BINARY)[1]
    thresh2 = cv.threshold(v, 25, 255, cv.THRESH_BINARY)[1]
    thresh2 = 255 - thresh2

    # combine the two threshold images as a mask
    mask = cv.add(thresh1,thresh2)

    # use mask to remove lines in background of input
    result = img.copy()
    result[mask==0] = (255,255,255)
    return result #np.concatenate((result, mask, thresh1, thresh2), axis=1)

def get_ax(rows=1, cols=1,figsize=(4,4), imgmode=True, returnfig=False):
    fig, axes = plt.subplots(figsize=figsize, dpi = 100, nrows=rows, ncols=cols)
    if imgmode:
        if rows == 1 and cols == 1:
            axes.clear()
            axes.get_xaxis().set_visible(False)
            axes.get_yaxis().set_visible(False)
        else:
            for ax in axes:
                if (isinstance(ax,np.ndarray)):
                    for a in ax:
                        a.clear()
                        a.get_xaxis().set_visible(False)
                        a.get_yaxis().set_visible(False)
                else:
                    ax.clear()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
    return (fig, axes) if returnfig else axes

def getpath():
    par_dir = "barrasThiago/"
    files = os.listdir(par_dir)
    index = np.random.randint(0, len(files))
    return par_dir + files[index]

def resize(img):
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # print(width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

def cluster(img, K=3):
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center = cv.kmeans(Z,K,None,criteria,5,cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((img.shape))

def formatSize(img):
    im = resize(img)
    im = im[bars_lim["y"][0]:bars_lim["y"][1], bars_lim["x"][0]:bars_lim["x"][1]]
    return im[bars_lim2["y"][0]:bars_lim2["y"][1],:]