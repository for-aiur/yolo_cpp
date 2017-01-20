#include <fstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <yolo.h>

int main(int argc, char** argv)
{
    if(argc < 2){
        fprintf(stderr, "usage: %s <videofile>\n", argv[0]);
        return 0;
    }

    std::ofstream file("out.csv");
    int frame_no = 0;
    char delimiter = ',';

    Yolo yolo;
    yolo.setConfigFilePath("cfg/yolo.cfg");
    yolo.setDataFilePath("cfg/coco.data");
    yolo.setWeightFilePath("data/yolo.weights");
    yolo.setAlphabetPath("data/labels/");
    yolo.setNameListFile("data/coco.names");

    cv::VideoCapture capture(argv[1]);
    if(!capture.isOpened())
    {
        std::cout << "cannot read video file" << std::endl;
        return 0;
    }

    cv::Mat img;
    while(true)
    {
        capture >> img;
        if(img.empty())
            break;

        //cv::resize(img, img, cv::Size(544,544));

        std::vector<DetectedObject> detection;
        yolo.detect(img, detection);

        for(int i = 0; i < detection.size(); i++)
        {
            DetectedObject& o = detection[i];
            cv::rectangle(img, o.bounding_box, cv::Scalar(255,0,0), 2);

            const char* class_name = yolo.getNames()[o.object_class];

            char str[255];
            //sprintf(str,"%s %f", names[o.object_class], o.prob);
            sprintf(str,"%s", class_name);
            cv::putText(img, str, cv::Point2f(o.bounding_box.x,o.bounding_box.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);

            if( strcmp(class_name,"person") == 0)
            {
                std::ostringstream oss;
                oss << frame_no << delimiter
                    << o.bounding_box.x << delimiter
                    << o.bounding_box.y << delimiter
                    << o.bounding_box.width << delimiter
                    << o.bounding_box.height;
                file << oss.str().c_str() << std::endl;
            }
        }

        cv::imshow("yolo++ demo", img);
        cv::waitKey(1);

        frame_no++;
    }
}
