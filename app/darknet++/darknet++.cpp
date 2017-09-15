#include <fstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <yolo.h>

void rotate_image_90n(cv::Mat &src, cv::Mat &dst, int angle)
{   
   if(src.data != dst.data){
       src.copyTo(dst);
   }

   angle = ((angle / 90) % 4) * 90;

   //0 : flip vertical; 1 flip horizontal
   bool const flip_horizontal_or_vertical = angle > 0 ? 1 : 0;
   int const number = std::abs(angle / 90);          

   for(int i = 0; i != number; ++i){
       cv::transpose(dst, dst);
       cv::flip(dst, dst, flip_horizontal_or_vertical);
   }
}

int main(int argc, char** argv)
{
    if(argc < 2){
        fprintf(stderr, "usage: %s <videofile>\n", argv[0]);
        return 0;
    }

    Yolo yolo;
    yolo.setConfigFilePath("cfg/tiny-yolo-tayse.cfg");
    yolo.setDataFilePath("cfg/tayse.data");
    yolo.setWeightFilePath("/home/yildirim/Dropbox/tayse/models/tiny-yolo-tayse_16000.weights");
    yolo.setAlphabetPath("data/labels/");
    yolo.setNameListFile("data/tayse.names");
    yolo.setThreshold(0.16);

    cv::VideoCapture capture(argv[1]);
    if(!capture.isOpened())
    {
        std::cout << "cannot read video file" << std::endl;
        return 0;
    }

//    cv::Mat img = cv::imread("/home/yildirim/Dropbox/tayse/deep/1/images/img_3.png");
    cv::Mat img;
    while(true)
    {
        capture >> img;
        if(img.empty())
            break;

	rotate_image_90n(img, img, 90);

        cv::resize(img, img, cv::Size(412,412));

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
        }

        cv::imshow("yolo++ demo", img);
        cv::waitKey(1);

    }
}
