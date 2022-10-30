cmake_minimum_required(VERSION 3.10)
project("GPU_RPMA")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O2 -mclwb")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "-O2")

set(RDMA_LIB_SOURCES gpu_direct_rdma_access.c utils.c)
set(PMEMDUMP_SOURCES pool.cpp object.cpp chksystem.cpp pmemdump.cpp)
set(SERVER_SOURCES pool.cpp object.cpp chksystem.cpp server.cpp)

add_library(gpu_rdma SHARED ${RDMA_LIB_SOURCES})
target_link_libraries(gpu_rdma rdmacm ibverbs mlx5)
add_executable(pmemdump ${PMEMDUMP_SOURCES})
add_executable(server ${SERVER_SOURCES})
target_link_libraries(server gpu_rdma rdmacm ibverbs mlx5)
