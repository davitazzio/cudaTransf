from __future__ import division
import cv2
import os
import numpy as np


def census(x, ndisp=256, wsize=5):
    image = x
    h,w = image.shape
    kernel_size = (wsize-1)//2  
    padded_x = np.pad(image, ((kernel_size,kernel_size),(kernel_size,kernel_size)), 'constant')
    cost = np.zeros((h,w), dtype='uint32')
    pos=0
    
    for u in range(h):
        for v in range(w):
            pos=0
            for off_u  in range(0, wsize):
                for off_v in range(0, wsize):
                    if off_u == off_v == kernel_size:
                        continue
                    if padded_x[u+off_u, v+off_v] > image[u,v]:
                        cost[u,v] = np.bitwise_xor(cost[u,v], 1<<pos)
                        pos=pos+1
                    else:
                        pos=pos+1
    return cost


if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread('example.png'), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (512,256))
    #result = cv2.cvtColor(cv2.imread('result.png'), cv2.COLOR_BGR2GRAY)
    print(img.shape)
    cost = census(img, wsize=5)
    print(cost.shape)

    cv2.imwrite('cost.png', cost.astype(np.uint8))


