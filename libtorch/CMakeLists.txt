cmake_minimum_required(VERSION 3.20)

project(libtorch_mnist_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------
# LibTorch
# ------------------------------------------------------------

set(CMAKE_PREFIX_PATH "C:/libtorch")

find_package(Torch REQUIRED)

# ------------------------------------------------------------
# Executable
# ------------------------------------------------------------

add_executable(
    libtorch_mnist_benchmark
    main.cpp
)

# ------------------------------------------------------------
# Link LibTorch
# ------------------------------------------------------------

target_link_libraries(
    libtorch_mnist_benchmark
    "${TORCH_LIBRARIES}"
)

# Required by LibTorch
set_property(
    TARGET libtorch_mnist_benchmark
    PROPERTY CXX_STANDARD 17
)