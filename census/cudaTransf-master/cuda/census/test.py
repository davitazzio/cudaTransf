'''
Census tests, similar to tests in Unflow by Meister et al https://arxiv.org/pdf/1711.07837.pdf
Code: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/test/ops/correlation.py
'''

import tensorflow as tf
import numpy as np
import os 
import sys
from tf_ops import census as cuda_census
from python.census import census as py_census
import cv2
np.set_printoptions(threshold=sys.maxsize)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

class CensusTest(tf.test.TestCase):
    def _test_census(self, in0, out, **kwargs):
        with self.test_session(use_gpu=True) as sess:
            in0 = np.expand_dims(in0, -1)
            in0_op = tf.constant(in0, tf.float32)
            result_op = cuda_census(in0_op, **kwargs)
            result = sess.run(result_op)
            result = np.squeeze(result)
            if out is not None:
                self.assertAllClose(out, result)

    def test_single_image_census(self):
        img = cv2.cvtColor(cv2.imread('python/example.png'), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (512, 256))
        expected = py_census(img, wsize=5)
        self._test_census(img, expected)

    
if __name__ == "__main__":
    tf.test.main()
