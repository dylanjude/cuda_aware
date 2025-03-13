#
# Mercury CMakeLists.txt file
#
cmake_minimum_required(VERSION 3.18)

project(MYTEST)

set(CMAKE_CUDA_FLAGS "-lineinfo -rdynamic --Wno-deprecated-gpu-targets -m64 -fPIC")
set(CMAKE_CUDA_ARCHITECTURES "60;70;80" CACHE STRING "CUDA archs")  
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
enable_language(CUDA)
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Project Build Type")

find_package(MPI REQUIRED)

file(GLOB srcs "${MYTEST_SOURCE_DIR}/*.cu")

add_executable(run ${srcs})
target_include_directories(run PUBLIC ${MPI_C_INCLUDE_PATH})
target_link_libraries(run ${MPI_LIBRARIES})
set_target_properties( run PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${MYTEST_BINARY_DIR})

