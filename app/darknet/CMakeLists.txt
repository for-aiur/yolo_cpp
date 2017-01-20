cmake_minimum_required(VERSION 2.8.11)

set(APP_NAME darknet)

file(GLOB APP_SOURCE *.h *.cpp)

include_directories(${YOLO_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(${APP_NAME} ${APP_SOURCE})
target_link_libraries(
        ${APP_NAME}
        ${DARKNET_CORE_LIBNAME}
        ${DARKNET_CUDA_LIBNAME}
        ${OpenCV_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CMAKE_THREAD_LIBS_INIT})
