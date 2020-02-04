#!/bin/bash
set -e 
echo "Building Sad Kernel"
cd cuda/sad
python tf_ops.py
echo "Cuda Sad Kernel compiled"
echo "Building census kernel"
cd census
python tf_ops.py
cd ..
echo "Cuda Census Kernel compiled"
echo "Building census_sad kernel"
cd census_sad
python tf_ops.py
cd ..
echo "Cuda Census_sad Kernel compiled"
echo "Running Sad tests"
python test.py
