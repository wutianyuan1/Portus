cmake_minimum_required(VERSION 3.10)
project("GPU_RPMA")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O2 -mclwb")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "-O2")

find_library(RDMACM_LIBRARIES rdmacm)
find_library(IBVERBS_LIBRARIES ibverbs)
find_library(MLX5_LIBRARIES mlx5)

include_directories(common server)
set(RDMA_LIB_SOURCES common/gpu_direct_rdma_access.c common/utils.c)
set(RPMA_SHARED_SOURCES server/pool.cpp server/object.cpp server/chksystem.cpp)
set(RPMACTL_SOURCES server/rpmactl.cpp)
set(MSERVER_SOURCES server/checkpointserver.cpp server/mserver.cpp)

add_library(rpma_common SHARED ${RPMA_SHARED_SOURCES})

add_library(gpu_rdma SHARED ${RDMA_LIB_SOURCES})
target_link_libraries(gpu_rdma ${RDMACM_LIBRARIES} ${IBVERBS_LIBRARIES} ${MLX5_LIBRARIES})

add_executable(rpmactl ${RPMACTL_SOURCES})
target_link_libraries(rpmactl rpma_common hdf5 hdf5_cpp)

add_executable(mserver ${MSERVER_SOURCES})
target_link_libraries(mserver gpu_rdma rpma_common ${RDMACM_LIBRARIES} ${IBVERBS_LIBRARIES} ${MLX5_LIBRARIES})
set_target_properties(mserver PROPERTIES COMPILE_FLAGS "-pthread" LINK_FLAGS "-pthread")
