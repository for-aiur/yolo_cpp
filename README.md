# yolo_cpp
C++ed version of Darknet

This is a repository for the developers who need to integrate yolo into their c++ projects. 
Only contribution to the original darknet code is a simple wrapper class around object detection functionality. 
You can immediately start building up a cpp project based on [YOLO](http://pjreddie.com/darknet) object detection. 

### What is Yolo and Darknet

Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

### Requirements
* OpenCV
* CUDA
* pthreads

As opposed to the original darknet, it is not possible to compile without CUDA and OpenCV.

### Usage

```sh
mkdir build
cd build
cmake ..
make 
```

What you get is the following executables in the source folder. 

* darknet -> This does the same job as the original darknet, but compiled with g++
* darknet_cpp -> This is a sample application demonstrates the usage of my yolo wrapper

You can copy all auxilary files from original darknet project and run example commands on the darknet website without any adjustment. 

If you want to develop a cpp application just add a new folder in /app folder, configure your cmake environment and there you go.
