'''
Correlation 3D tests, based on 2D test suite of Unflow by Meister et al https://arxiv.org/pdf/1711.07837.pdf
Code: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/test/ops/correlation.py
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import gradient_checker
import os 
import sys
sys.path.append('../kernels')
from ops import correlation3D

class CorrelationTest(tf.test.TestCase):
    def _test_correlation(self, in0, in1, out=None, **kwargs):
        with self.test_session(use_gpu=True) as sess:
            in0_op = tf.constant(in0, tf.float32)
            in1_op = tf.constant(in1, tf.float32)
            result_op = correlation2D(in0_op, in1_op, **kwargs)
            result = sess.run(result_op)

            if out is not None:
                self.assertAllClose(out, result)

            jacob_t, jacob_n = gradient_checker.compute_gradient([in0_op, in1_op],
                                                                 [in0.shape, in1.shape],
                                                                 result_op, result.shape)
            self.assertAllClose(jacob_t, jacob_n, 1e-3, 1e-3)

    def test_correlation_trivial(self):
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
        self._test_correlation(first, second, expected,
                               kernel_size=1, stride_2=1, max_displacement=0,
                               pad=0)

    def test_correlation_batch(self):
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

    def test_correlation_channels(self):
        pass

    def test_correlation_3x3(self):
        first = [
          [1, 1, 3],
          [0, 0, 1],
          [2, 2, 0.2]]
        second = [
          [1, 2, 0.1],
          [3, 4, 2.2],
          [4, 5, 1.6]]

        first = np.reshape(first, [1, 1, 3, 3])
        second = np.reshape(second, [1, 1, 3, 3])
        self._test_correlation(first, second, None,
                             kernel_size=3, stride_2=1, max_displacement=1,
                             pad=2)

def naive_correlation3D(input_a, input_b, **kwargs):
        ''' Correlation 3D op in pure python 
        Params:
            input_a, input_b: BxHxWxC input tensors
            kwargs: dict with params
                - stride: horizontal, vertical and depth stride [default=1]
                - k: kernel size [default=3]
        '''
        assert input_a.shape == input_b.shape
        assert input_a.ndim == 4
        k = kwargs.get('k', 3)
        stride = kwargs.get('stride', 1)
        b, c, h, w = input_a.shape
        positions = _generate_coordinates(h,w)
        padded_b = _pad_input(input_b, k)
        for (row, col) in positions:
            # iterate over kernel
            pass

def _generate_coordinates(height, width):
    ''' Generate coordinates to explore the volume 
        Params: 
            height, width: int, height and width of space respectively
        Returns:
            xv,yv: horizontal and vertical search coordinates
    '''
    xv, yv = np.meshgrid(np.arange(0, width, 1), np.arange(0, height, 1))
    grid = np.array([xv,yv])
    positions = grid.T.reshape(-1,2)
    return positions

def _pad_input(x, k, **kwargs):
    ''' Pad the input, so the correlation do not fail near borders
        Params:
            x: input to pad with shape BxCxHxW
            k: kernel size
            kwargs: dict with arguments
                padding_value: constant value to use during padding [default=0]
        Returns:
            output: input that has been padded along C,H,W by k.
    '''
    assert k % 2 !=0
    kernel_radius = int((k-1)/2)
    padding_value = kwargs.get('padding_value', 0)
    padding_shape = ((0, 0), (kernel_radius, kernel_radius), (kernel_radius, kernel_radius), (kernel_radius, kernel_radius))
    output = np.pad(x, padding_shape, 'constant', constant_values=padding_value)
    return output

def test_naive_correlation3D():
    ''' Testing naive correlation 3D '''
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
    naive_correlation3D(first, second)


if __name__ == "__main__":
    #tf.test.main()
    test_naive_correlation3D()