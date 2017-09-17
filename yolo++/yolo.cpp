#include "yolo.h"
#include <stdexcept>

#include <opencv2/highgui/highgui.hpp>

YoloPython::YoloPython() : w(412), h(412), isLoaded(false){
	yolo = new Yolo();
}

void YoloPython::init(std::string weight, std::string names, std::string cfg){
	if(!isLoaded){
   		yolo->setConfigFilePath(cfg.c_str());
    		yolo->setWeightFilePath(weight.c_str());
    		yolo->setNameListFile(names.c_str());
		isLoaded=true;
	}
}
 
void YoloPython::setImgSize(int w, int h) {
	this->w = w;
	this->h = h;
}

void YoloPython::setThreshold(float val){
	yolo->setThreshold(val);
}

int YoloPython::detect(std::string path, int cls) {
                cv::Mat img = cv::imread(path.c_str());
		cv::resize(img, img, cv::Size(w,h));
		std::vector<DetectedObject> all_objects;
                yolo->detect(img, all_objects);
		
		std::vector<DetectedObject> objects_local;
		for(auto obj : all_objects){
			if (obj.object_class == cls)
				objects_local.push_back(obj);
		}
		objects.swap(objects_local);
                return objects.size();
}

float YoloPython::getComponent(int bb_idx, int component_idx) 
{
	if( bb_idx < 0 || bb_idx >= objects.size())
		return 0.0;

	if( component_idx == 0 )
		return objects[bb_idx].bounding_box.x;
	if( component_idx == 1 )
		return objects[bb_idx].bounding_box.y;
	if( component_idx == 2 )
		return objects[bb_idx].bounding_box.width;
	if( component_idx == 3 )
		return objects[bb_idx].bounding_box.height;
	return 0.0f;
}

Yolo::Yolo() : thresh(.24f),
               hier_thresh(.5f),
               nms(0.4f),
               datacfg(NULL),
               cfgfile(NULL),
               weightfile(NULL),
               alphabet_path(NULL),
               names(NULL),
               alphabet(NULL),
               name_list(NULL)
{

}

void Yolo::ocv_to_yoloimg(const cv::Mat& img, image& yolo_img)const
{
    yolo_img = make_image(img.size().width, img.size().height, img.channels());
    unsigned char *data = img.data;
    int h = img.size().height;
    int w = img.size().width;
    int c = img.channels();
    int step = w*c;
    int i, j, k, count=0;

    for(k= 0; k < c; ++k){
        for(i = 0; i < h; ++i){
            for(j = 0; j < w; ++j){
                yolo_img.data[count++] = data[i*step + j*c + k]/255.;
            }
        }
    }
}

void Yolo::setConfigFilePath(const char* filename)
{
    cfgfile = filename;
    net = parse_network_cfg(const_cast<char*>(cfgfile));
}

void Yolo::setDataFilePath(const char* filename)
{
    datacfg = filename;
}

void Yolo::setWeightFilePath(const char* filename)
{
    if(!net.n)
    {
        throw std::runtime_error("Can't load weights, there is a problem with network generation");
    }

    weightfile = filename;
    if(weightfile){
        load_weights(&net, const_cast<char*>(weightfile));
    }
    set_batch_network(&net, 1);
}

void Yolo::setThreshold(const float thresh)
{
    this->thresh = thresh;
}

void Yolo::setHierThreshold(const float thresh)
{
    hier_thresh = thresh;
}

void Yolo::setAlphabetPath(const char* filename)
{
    alphabet_path = filename;
    alphabet = load_alphabet_custom(alphabet_path);
}

void Yolo::setNms(const float nms)
{
    this->nms = nms;
}

void Yolo::setNameListFile(const char* filename)
{
    name_list = filename;
    names = get_labels(const_cast<char*>(name_list));
}

char** Yolo::getNames()
{
    return names;
}

float Yolo::getThreshold()
{
    return thresh;
}

void Yolo::detect(const cv::Mat& img, std::vector<DetectedObject>& detection)const
{
    image im, sized;
    float **probs = NULL;
    cv::Mat img_local;

    try {
        cv::cvtColor(img, img_local, cv::COLOR_BGR2RGB);

        ocv_to_yoloimg(img_local, im);
        sized = resize_image(im, net.w, net.h);

        layer l = net.layers[net.n - 1];
        int output_size = l.w * l.h * l.n;

        box boxes[output_size];
        probs = new float *[output_size]();
        for (int i = 0; i < output_size; i++)
            probs[i] = new float[l.classes + 1]();

        network_predict(net, sized.data);

        get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);

        if (l.softmax_tree && nms)
            do_nms_obj(boxes, probs, output_size, l.classes, nms);
        else if (nms)
            do_nms_sort(boxes, probs, output_size, l.classes, nms);

        //draw_detections(im, output_size, thresh, boxes, probs, names, alphabet, 20);
        //show_image(im, "predictions");
        //cv::waitKey(1);

        for (int i = 0; i < output_size; i++) {
            int most_probable_class_index = max_index(probs[i], l.classes);
            float prob = probs[i][most_probable_class_index];
            if (prob > thresh) {
                box &b = boxes[i];
                int left = (b.x - b.w / 2.) * im.w;
                int right = (b.x + b.w / 2.) * im.w;
                int top = (b.y - b.h / 2.) * im.h;
                int bot = (b.y + b.h / 2.) * im.h;
                if (left < 0) left = 0;
                if (right > im.w - 1) right = im.w - 1;
                if (top < 0) top = 0;
                if (bot > im.h - 1) bot = im.h - 1;

                cv::Rect r(left, top, std::fabs(left - right), std::fabs(top - bot));
                detection.push_back(DetectedObject(most_probable_class_index, prob * 100., r));
            }
        }
        delete[] probs;
        free_image(im);
        free_image(sized);
    }
    catch(...)
    {
        if(probs)
            delete[] probs;

        free_image(im);
        free_image(sized);
        throw std::runtime_error("Yolo related error");
    }

}
