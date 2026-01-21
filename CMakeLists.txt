cmake_minimum_required (VERSION 3.19 FATAL_ERROR) # 3.19 is required by cutlass

project (lasagna CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
else()
  set(CMAKE_CUDA_ARCHITECTURES "80-real;90-real")
endif()

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

add_library(ovlp SHARED
				overlap.cu
)

set_target_properties(ovlp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
        CUDA_SEPARABLE_COMPILATION ON
)
