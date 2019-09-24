from __future__ import division
import cv2
import os
import numpy as np


def census(x, ndisp=256, wsize=9):
    ''' Easy implementation of census transform'''
    h,w = x.shape
    kernel_size = (wsize-1)//2  
    padded_x = np.pad(x, ((kernel_size,kernel_size),(kernel_size,kernel_size)), 'constant')
    assert padded_x.shape[0] == (h+2*kernel_size)
    assert padded_x.shape[1] == (w+2*kernel_size)
    census = np.zeros((h,w), dtype='uint8')

    for u in range(kernel_size, h+kernel_size):
        for v in range(kernel_size, w+kernel_size):
            for off_u  in range(-kernel_size, kernel_size+1):
                for off_v in range(-kernel_size, kernel_size+1):
                    if off_u == off_v == 0:
                        continue
                    census[u-kernel_size,v-kernel_size] += 1 if padded_x[u+off_u, v+off_v] >= padded_x[u-kernel_size,v-kernel_size] else 0
    return census

if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread('example.png'), cv2.COLOR_BGR2GRAY)
    result = cv2.cvtColor(cv2.imread('result.png'), cv2.COLOR_BGR2GRAY)
    print(result)
    cost = census(img)
    cv2.imwrite('cost.png', cost)