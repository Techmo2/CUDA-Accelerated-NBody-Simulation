project("Cuda Accelerated NBody Simulation")

cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -ccbin /usr/bin/g++-6
    )

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    simulation
    src/simulation.cu
    )