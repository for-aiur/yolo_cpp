#ifndef DARKNET_OBJECT_H
#define DARKNET_OBJECT_H

#include <opencv2/imgproc/imgproc.hpp>

struct DetectedObject
{
    int object_class;
    float prob;
    cv::Rect bounding_box;

    DetectedObject() : object_class(-1), prob(0.), bounding_box(cv::Rect(0,0,0,0)){}
    DetectedObject(int object_class, float prob, cv::Rect bb) : object_class(object_class), prob(prob), bounding_box(bb){}
};

#endif //DARKNET_OBJECT_H
