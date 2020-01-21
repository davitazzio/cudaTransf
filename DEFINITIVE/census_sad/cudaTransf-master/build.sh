#!/bin/bash
set -e 
echo "Building CensusSad Kernel"
cd cuda/census_sad
python tf_ops.py
echo "Cuda CensusSad Kernel compiled"
echo "Running CensusSad tests"
python test.py
