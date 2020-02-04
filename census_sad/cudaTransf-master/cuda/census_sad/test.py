'''
Census tests, similar to tests in Unflow by Meister et al https://arxiv.org/pdf/1711.07837.pdf
Code: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/test/ops/correlation.py
'''

import tensorflow as tf
import numpy as np
import os 
import sys
from tf_ops import census_sad as cuda_census_sad
from python.census import census as py_census
from python.sad import sad as py_sad
from python.sad import min as py_argmin
import cv2
np.set_printoptions(threshold=sys.maxsize)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

class CensusSadTest(tf.test.TestCase):
    def _test_census_sad_function(self, in0, in1, out, **kwargs):

        with self.test_session(use_gpu=True) as sess:
            
            in0 = np.expand_dims(in0, -1)
            in0_op = tf.constant(in0, tf.float32)
            
            in1 = np.expand_dims(in1, -1)
            in1_op = tf.constant(in1, tf.float32)
            
            result_op = cuda_census_sad(in0_op, in1_op, wsize=5, ndisp=128, **kwargs)
            result = sess.run(result_op)
            result = np.squeeze(result)
            if out is not None:
                self.assertAllClose(out, result)

    def test_census_sad(self):
        
        img_left = cv2.cvtColor(cv2.imread('python/scene1.row3.col1.ppm'), cv2.COLOR_BGR2GRAY)
        img_left = cv2.resize(img_left, (64,32))

        img_right = cv2.cvtColor(cv2.imread('python/scene1.row3.col5.ppm'), cv2.COLOR_BGR2GRAY)
        img_right = cv2.resize(img_right, (64,32))

        py_census_left= py_census(img_left, wsize=5)
        py_census_right=py_census(img_right, wsize=5)
        py_result=py_sad(py_census_left, py_census_right, ndisp=128)
        expected=py_argmin(py_result)

        self._test_census_sad_function(img_left, img_right, expected)
    
if __name__ == "__main__":
    tf.test.main()
