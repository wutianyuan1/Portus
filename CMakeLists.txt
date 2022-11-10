cmake_minimum_required(VERSION 3.10)
project("GPU_RPMA")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O2 -mclwb")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "-O2")

set(RDMA_LIB_SOURCES gpu_direct_rdma_access.c utils.c)
set(RPMACTL_SOURCES pool.cpp object.cpp chksystem.cpp rpmactl.cpp)
set(MSERVER_SOURCES pool.cpp object.cpp chksystem.cpp checkpointserver.cpp mserver.cpp)

add_library(gpu_rdma SHARED ${RDMA_LIB_SOURCES})
target_link_libraries(gpu_rdma rdmacm ibverbs mlx5)

add_executable(rpmactl ${RPMACTL_SOURCES})
target_link_libraries(rpmactl hdf5 hdf5_cpp)

add_executable(mserver ${MSERVER_SOURCES})
target_link_libraries(mserver gpu_rdma rdmacm ibverbs mlx5)
set_target_properties(mserver PROPERTIES COMPILE_FLAGS "-pthread" LINK_FLAGS "-pthread")
