#ifndef DARKNET_YOLO_H
#define DARKNET_YOLO_H

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>

#include <parser.h>
#include <utils.h>
#include <option_list.h>
#include <region_layer.h>
#include <image.h>

#include <object.h>

class Yolo {
public:

    Yolo();

    void detect(const cv::Mat& img, std::vector<DetectedObject>& detection)const; //Throws

    void setConfigFilePath(const char* filename);
    void setDataFilePath(const char* filename);
    void setWeightFilePath(const char* filename);
    void setNameListFile(const char* filename);
    void setThreshold(const float thresh);
    void setHierThreshold(const float thresh);
    void setAlphabetPath(const char* filename);
    void setNms(const float nms);

    char** getNames();

private:
    void ocv_to_yoloimg(const cv::Mat& img, image& yolo_img)const;

    const char *datacfg;
    const char *cfgfile;
    const char *weightfile;
    const char* alphabet_path;
    const char* name_list;

    float thresh;
    float hier_thresh;
    float nms;

    //members
    network net;
    char **names;
    image **alphabet;

    //Avoid copy by keeping these unimplemented
    const Yolo& operator=(const Yolo& rhs);
    Yolo(const Yolo& copy);
};

#endif //DARKNET_YOLO_H
