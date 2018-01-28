#!/bin/bash
rm -r build
mkdir build
cd build
cmake ..
cmake --build .
echo "Build complete. Program 'simulation' is located in the 'build' directory."
