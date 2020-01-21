from __future__ import division
import cv2
import os
import numpy as np


def sad(x, y, ndisp=32):
    image = x
    h,w = image.shape
    cost = np.full((h, w, ndisp), 64, dtype="float")
    for u in range(h):
        for v in range(w):
            for d in range(0, ndisp):
                if (v-d)>=0:
                    r= np.bitwise_xor(x[u,v], y[u, v-d])
                    count="{0:b}".format(r).count("1")
                    cost[u,v,d]=count       
    return cost

def min(x):
    h,w,ndisp=x.shape
    disparity = np.zeros((h,w), dtype='float')

    for u in range(h):
        for v in range(w):
            min=x[u, v, 0]
            index=0
            for d in range(1, ndisp):
                if x[u, v, d]<min:
                    min=x[u,v,d]
                    index=d
            disparity[u, v]= index
    return disparity


if __name__ == '__main__':
    img1 = cv2.cvtColor(cv2.imread('scene1.row3.col1.ppm'), cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img1, (64,32))
    img2 = cv2.cvtColor(cv2.imread('scene1.row3.col5.ppm'), cv2.COLOR_BGR2GRAY)
    img2 = cv2.resize(img2, (64,32))
    cost = sad(img1, img2, ndisp=32)
    disparity=min(cost)
    cv2.imwrite('cost.png', disparity.astype(np.uint8))

