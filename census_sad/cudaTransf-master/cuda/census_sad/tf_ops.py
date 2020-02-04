'''
Adapted from Meister Unflow
https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/ops.py
'''

import os
import sys
import tensorflow as tf
import subprocess
from tensorflow.python.framework import ops

# Register ops for compilation here
OP_NAMES = ['census_sad']

cwd = os.getcwd()
os.chdir(os.path.dirname(os.path.realpath(__file__)))

config = {}
config['g++'] = 'g++'

def compile(op=None):
    if op is not None:
        to_compile = [op]
    else:
        to_compile = OP_NAMES
    
    tf_inc = " ".join(tf.sysconfig.get_compile_flags())
    tf_lib = " ".join(tf.sysconfig.get_link_flags())
    for n in to_compile:
        base = n + "_op"
        fn_cu_cc = base + ".cu.cc"
        fn_cu_o = base + ".cu.o"
        fn_cc = base + ".cc"
        fn_o = base + ".o"
        fn_so = base + ".so"

        out, err = subprocess.Popen(['which', 'nvcc'], stdout=subprocess.PIPE).communicate()
        cuda_dir = out.decode().split('/cuda')[0]

        nvcc_cmd = "nvcc -std=c++11 -c -o {} {} {} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -I " + cuda_dir + " --expt-relaxed-constexpr -w"
        nvcc_cmd = nvcc_cmd.format(" ".join([fn_cu_o, fn_cu_cc]), tf_inc, tf_lib)
        subprocess.check_output(nvcc_cmd, shell=True)
        
        gcc_cmd = "{} -std=c++11 -shared -o {} {} -fPIC -L " + cuda_dir + "/cuda/lib64 -lcudart {} -O2 -D GOOGLE_CUDA=1"
        gcc_cmd = gcc_cmd.format(config['g++'],	" ".join([fn_so, fn_cu_o, fn_cc]), tf_inc, tf_lib)
        subprocess.check_output(gcc_cmd, shell=True)

if __name__ == "__main__":
    compile()

module = sys.modules[__name__]
for n in OP_NAMES:
    lib_path = './{}_op.so'.format(n)
    try:
        op_lib = tf.load_op_library(lib_path)
    except:
        compile(n)
        op_lib = tf.load_op_library(lib_path)
    setattr(module, '_' + n + '_module', op_lib)


os.chdir(cwd)

def census_sad(left, right, **kwargs):
    return _census_sad_module.census_sad(left, right, **kwargs)[0]
