from __future__ import division
import cv2
import os
import numpy as np


def census(x, ndisp=256, wsize=9):
    ''' Easy implementation of census transform'''
    h,w = x.shape
    kernel_size = (wsize-1)//2  
    padded_x = np.pad(x, ((kernel_size,kernel_size),(kernel_size,kernel_size)), 'constant')
    cost = np.zeros((h,w), dtype='uint8')

    for u in range(h):
        for v in range(w):
            for off_u  in range(-kernel_size, kernel_size+1):
                for off_v in range(-kernel_size, kernel_size+1):
                    if off_u == off_v == 0:
                        continue
                    cost[u,v] = (cost[u, v] << 1) | (1 if padded_x[u+off_u, v+off_v] >= x[u,v] else 0)
    return cost

if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread('example.png'), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (512,256))
    result = cv2.cvtColor(cv2.imread('result.png'), cv2.COLOR_BGR2GRAY)
    cost = census(img, wsize=5)
    cv2.imwrite('cost.png', cost)