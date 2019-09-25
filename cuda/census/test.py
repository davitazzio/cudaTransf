'''
Census tests, similar to tests in Unflow by Meister et al https://arxiv.org/pdf/1711.07837.pdf
Code: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/test/ops/correlation.py
'''

import tensorflow as tf
import numpy as np
import os 
import sys
from tf_handler import census as cuda_census
from python.census import census as py_census
import cv2

class CensusTest(tf.test.TestCase):
    def _test_census(self, in0, out=None, **kwargs):
        with self.test_session(use_gpu=True) as sess:
            in0 = np.expand_dims(in0, -1)
            in0_op = tf.constant(in0, tf.float32)
            result_op = cuda_census(in0_op, **kwargs)
            result = sess.run(result_op)

            if out is not None:
                self.assertAllClose(out, result)

    def test_single_image_census(self):
        img = cv2.cvtColor(cv2.imread('census/python/example.png'), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256,128))
        expected = py_census(img, wsize=9)
        self._test_census(img, expected)

    def _test_correlation_batch(self):
        first = [
           [1, 1, 2, 2],
           [0, 0, 2, 2],
           [3, 3, 4, 4],
           [3, 3, 2, 2]]
        second = [
           [1, 1, 2, 2],
           [0, 0, 2, 2],
           [3, 3, 4, 4],
           [3, 3, 2, 2]]

        first = np.reshape(first, [1, 1, 4, 4])
        second = np.reshape(second, [1, 1, 4, 4])
        expected = np.square(first)

        self._test_correlation(np.concatenate([first, first], 0),
                              np.concatenate([second, second], 0),
                              np.concatenate([expected, expected], 0),
                              kernel_size=1, stride_2=1, max_displacement=0,
                              pad=0)

if __name__ == "__main__":
    tf.test.main()