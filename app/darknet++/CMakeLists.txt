cmake_minimum_required(VERSION 2.8.11)

include_directories(${DARKNET_WRAPPER_INCLUDE_DIR})
include_directories(${YOLO_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})

find_package(PythonLibs 2.7 REQUIRED)
find_package(Boost COMPONENTS python REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR})

add_executable(darknet_cpp darknet++.cpp)
target_link_libraries(
        darknet_cpp
        ${DARKNET_CORE_LIBNAME}
        ${DARKNET_CUDA_LIBNAME}
        ${DARKNET_WRAPPER_LIBRARIES}
	python2.7
	${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CMAKE_THREAD_LIBS_INIT}
)
