#!/bin/bash

# Clean previous builds
rm -rf build dist *.so *.c

# Build the Cython extension
python3.12 setup.py build_ext --inplace