import matplotlib.pyplot as plt

import numpy as np
import cv2 as cv

from utils import get_ax, cluster, resize, smoothImage, removeGreyBackground
import utils as u

def getContours(im_raw):
    # thresh = img.copy()
    img = np.copy(im_raw)
    ret,thresh = cv.threshold(cv.cvtColor(img, cv.COLOR_BGR2GRAY),200,255,
        cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    contours,hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1)
    # for cnt in contours:
    #     img = cv.drawContours(img,[cnt],0,(0,255,0),2)
    
    # height, width, _ = img.shape
    # min_x, min_y = width, height
    # max_x = max_y = 0
    contours_minmax = []
    for cnt in contours:
        (x,y,w,h) = cv.boundingRect(cnt)
        # min_x, max_x = min(x, min_x), max(x+w, max_x)
        # min_y, max_y = min(y, min_y), max(y+h, max_y)
        
        img = cv.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0), 2)
        contours_minmax.append([[x, y], [x+w, y+h]])

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img2 = np.zeros_like(img)
    img2[:,:,0] = gray
    img2[:,:,1] = gray
    img2[:,:,2] = gray

    return img, contours, contours_minmax #np.concatenate((img, img2), axis=1) #thresh

def getBars(im):
    dir_par = "preprocessing/"
    img = resize(im)
    cv.imwrite(dir_par + "0-original.jpg", img)
    img = img[u.bars_lim["y"][0]:u.bars_lim["y"][1], u.bars_lim["x"][0]:u.bars_lim["x"][1]]
    cv.imwrite(dir_par + "0-cut.jpg", img)
    img = smoothImage(img)
    cv.imwrite(dir_par + "2-smooth.jpg", img)
    img = removeGreyBackground(img)
    cv.imwrite(dir_par + "3-removebackground.jpg", img)
    img = cluster(img)
    cv.imwrite(dir_par + "4-cluster.jpg", img)

    img = img[u.bars_lim2["y"][0]:u.bars_lim2["y"][1],:]
    img, contours, contours_minmax = getContours(img)
    cv.imwrite(dir_par + "5-contours.jpg", img)

    return img, contours, np.array(contours_minmax)


def saveBars():
    # img = cv.imread(getpath())
    img = cv.imread("barrasThiago/vermelhoAltura100.jpg")
    im = img.copy()
    img, bars_cnts, c_mm = getBars(img.copy())

    bars = c_mm[:-1]
    # np.save('bars_original', bars)

if __name__ == '__main__':

    # size = 4
    # axes = get_ax(2,int(size/2))
    # for i in range(0,size):
    #     print(getpath(), int(i/(size/2)), int(i%(size/2)))
    #     img = cv.imread(getpath())
    #     bars = getBars(img.copy())
        
    #     axes[int(i/(size/2)), int(i%(size/2))].imshow(np.concatenate((cv.resize(img, (700-110,300)),bars), axis=1)) 

    # imgplot = plt.imshow(subimg)    
    # plt.show()

    saveBars()
    

# cv.imshow('res2',opening)
# cv.waitKey(0)
# cv.destroyAllWindows()