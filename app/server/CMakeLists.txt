cmake_minimum_required(VERSION 2.8.11)

include_directories(${DARKNET_WRAPPER_INCLUDE_DIR})
include_directories(${YOLO_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(server echoserver.h echoserver.cpp main.cpp)
target_link_libraries(
        darknet_cpp
        ${DARKNET_CORE_LIBNAME}
        ${DARKNET_CUDA_LIBNAME}
        ${DARKNET_WRAPPER_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CMAKE_THREAD_LIBS_INIT}
)




