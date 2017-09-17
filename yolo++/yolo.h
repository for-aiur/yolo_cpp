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
#include <opencv2/highgui/highgui.hpp>

#include <boost/python/numeric.hpp>

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

    float getThreshold();

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

struct YoloPython{
	YoloPython();
	void init(std::string weight, std::string names, std::string cfg);
	void setImgSize(int w, int h);
	void setThreshold(float val);
	int detect(std::string path, int cls);
	float getComponent(int bb_idx, int component_idx);
	Yolo* yolo;
        std::vector<DetectedObject> objects;
	int w,h;
	bool isLoaded;
};


#endif //DARKNET_YOLO_H
