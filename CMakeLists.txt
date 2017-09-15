cmake_minimum_required(VERSION 2.8.11)

#Suppressing CMAKE 3.0 warnings
if(POLICY CMP0043)
    cmake_policy(SET CMP0037 OLD)
endif()

add_definitions(-std=c++11)

set(PROJECT_NAME darknet)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/)

add_definitions(-DOPENCV)
add_definitions(-DGPU)

project(${PROJECT_NAME})

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

add_subdirectory(src)
add_subdirectory(yolo++)
add_subdirectory(app)
