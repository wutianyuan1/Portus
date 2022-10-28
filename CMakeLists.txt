cmake_minimum_required(VERSION 3.10)
project("GPU_RPMA")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O2 -mclwb")
set(SOURCES pool.cpp object.cpp chksystem.cpp test-chksys.cpp)

add_executable("test" ${SOURCES})
