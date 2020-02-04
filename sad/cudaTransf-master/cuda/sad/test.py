'''
Census tests, similar to tests in Unflow by Meister et al https://arxiv.org/pdf/1711.07837.pdf
Code: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/test/ops/correlation.py
'''

import tensorflow as tf
import numpy as np
import os 
import sys
from tf_ops import sad as cuda_sad
from census.tf_ops import census
from census_sad.tf_ops import census_sad

import cv2
np.set_printoptions(threshold=sys.maxsize)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

class SadTest(tf.test.TestCase):
    def _test_sad_function(self, in0, in1, out, **kwargs):

        with self.test_session(use_gpu=True) as sess:
                        
            result_op = cuda_sad(in0, in1, ndisp=256, **kwargs)
            result = sess.run(result_op)
            if out is not None:
                self.assertAllClose(out, result)

    def test_sad(self):
        
        img_left = cv2.cvtColor(cv2.imread('images/scene1.row3.col1.ppm'), cv2.COLOR_BGR2GRAY)
        #img_left = cv2.resize(img_left, (64,32))

        img_right = cv2.cvtColor(cv2.imread('images/scene1.row3.col5.ppm'), cv2.COLOR_BGR2GRAY)
        #img_right = cv2.resize(img_right, (64,32))

        img_left = np.expand_dims(img_left, -1)
        img_left_op = tf.constant(img_left, tf.float32)
        img_right = np.expand_dims(img_right, -1)
        img_right_op = tf.constant(img_right, tf.float32)
        cuda_census_left= census(img_left_op, wsize=13)
        cuda_census_right=census(img_right_op, wsize=13)
        expected=census_sad(img_left, img_right, ndisp=256)

        self._test_sad_function(cuda_census_left, cuda_census_right, expected)
    
if __name__ == "__main__":
    tf.test.main()
