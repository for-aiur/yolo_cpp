cmake_minimum_required(VERSION 2.8.11)

set(MODULENAME ${PROJECT_NAME}++)

include_directories(${YOLO_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(DARKNET_WRAPPER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "${MODULENAME}: include dir" FORCE)
set(DARKNET_WRAPPER_LIBRARIES ${MODULENAME} CACHE INTERNAL "${MODULENAME}: library" FORCE)

file(GLOB YOLO_WRAPPER_SOURCE *.h *.cpp)
add_library(${MODULENAME} SHARED ${YOLO_WRAPPER_SOURCE})
target_link_libraries(
        ${MODULENAME}
        ${DARKNET_CORE_LIBNAME}
        ${DARKNET_CUDA_LIBNAME}
        ${OpenCV_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${CMAKE_THREAD_LIBS_INIT})

find_package(PythonLibs 2.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
find_package(Boost COMPONENTS python REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
add_library(yolopy SHARED yolopy.cpp)
target_link_libraries(yolopy ${Boost_LIBRARIES} ${MODULENAME})
set_target_properties(yolopy PROPERTIES PREFIX "")




