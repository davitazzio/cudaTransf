#!/bin/bash
set -e 
echo "Buildin Census Kernel"
cd cuda/census
python tf_ops.py
echo "Cuda Census Kernel compiled"
echo "Running Census tests"
python test.py