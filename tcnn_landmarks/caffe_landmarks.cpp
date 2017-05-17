#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace cv::dnn;
using namespace std;


int main(int argc, char **argv)
{
    cv::dnn::initModule();  //Required if OpenCV is built as static libs

    String modelTxt = "vanilla_deplay.prototxt";
    String modelBin = "vanillaCNN.caffemodel";
    String imageFile = (argc > 1) ? argv[1] : "face.png";

    Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
    if (net.empty())
    {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "prototxt:   " << modelTxt << std::endl;
        std::cerr << "caffemodel: " << modelBin << std::endl;
        exit(-1);
    }
    Mat img = imread(imageFile);
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << imageFile << std::endl;
        exit(-1);
    }

    resize(img, img, Size(40,40));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    Mat inputBlob = blobFromImage(img);   //Convert Mat to dnn::Blob batch of images
    net.setBlob(".data", inputBlob);        //set the network input
    net.forward();                          //compute output
    Mat probMat = net.getBlob("Dense2");   //gather output of "prob" layer
    cerr << probMat << endl;

    for (int i=0; i<5; i++) {
        int x = (probMat.at<float>(0,i*2  ) + 0.5) * 40;
        int y = (probMat.at<float>(0,i*2+1) + 0.5) * 40;
        circle(img, Point(x,y), 2, Scalar(255,0,0), 1);
    }
    namedWindow("LANDMARKS !",0);
    imshow("LANDMARKS !", img);
    waitKey();
    return 0;
}
