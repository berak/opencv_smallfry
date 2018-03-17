
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/optflow.hpp"

using namespace std;
using namespace cv;
using namespace optflow;


Mat flowToDisplay(const Mat &flow)
{
    Mat flow_split[2];
    Mat magnitude, angle;
    Mat hsv_split[3], hsv, rgb;
    split(flow, flow_split);
    cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);
    normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);
    hsv_split[0] = angle; // already in degrees - no normalization needed
    hsv_split[1] = Mat::ones(angle.size(), angle.type());
    hsv_split[2] = magnitude;
    merge(hsv_split, 3, hsv);
    cvtColor(hsv, rgb, COLOR_HSV2BGR);
    return rgb;
}


double computeOptFlow(const Mat &im1_, const Mat &im2_, const string &optflow_method = "DISflow_medium")
{
    double dist = -1.0;
    Mat_<Point2f> flow;
    Mat im1 = im1_.clone();
    Mat im2 = im2_.clone();

    if ( !im1.data || !im2.data )
    {
        printf("No image data \n");
        return -1;
    }
    if ( im1.size() != im2.size() || im1.channels() != im2.channels() )
    {
        printf("Dimension mismatch between input images\n");
        return -1;
    }
    // 8-bit images expected by all algorithms
    if ( im1.depth() != CV_8U )
        im1.convertTo(im1, CV_8U);
    if ( im2.depth() != CV_8U )
        im2.convertTo(im2, CV_8U);

    if ( (optflow_method == "farneback" || optflow_method == "tvl1" || optflow_method == "deepflow"
          || optflow_method == "DISflow_ultrafast" || optflow_method == "DISflow_fast"
          || optflow_method == "DISflow_medium") && im1.channels() == 3 )
    {   // 1-channel images are expected
        cvtColor(im1, im1, COLOR_BGR2GRAY);
        cvtColor(im2, im2, COLOR_BGR2GRAY);
    } else if ( optflow_method == "simpleflow" && im1.channels() == 1 )
    {   // 3-channel images expected
        cvtColor(im1, im1, COLOR_GRAY2BGR);
        cvtColor(im2, im2, COLOR_GRAY2BGR);
    }

    flow = Mat(im1.size[0], im1.size[1], CV_32FC2);
    Ptr<DenseOpticalFlow> algorithm;

    if ( optflow_method == "farneback" )
        algorithm = createOptFlow_Farneback();
    else if ( optflow_method == "simpleflow" )
        algorithm = createOptFlow_SimpleFlow();
    else if ( optflow_method == "tvl1" )
        algorithm = createOptFlow_DualTVL1();
    else if ( optflow_method == "deepflow" )
        algorithm = createOptFlow_DeepFlow();
    else if ( optflow_method == "sparsetodenseflow" )
        algorithm = createOptFlow_SparseToDense();
    else if ( optflow_method == "pcaflow" )
        algorithm = createOptFlow_PCAFlow();
    else if ( optflow_method == "DISflow_ultrafast" )
        algorithm = createOptFlow_DIS(DISOpticalFlow::PRESET_ULTRAFAST);
    else if (optflow_method == "DISflow_fast")
        algorithm = createOptFlow_DIS(DISOpticalFlow::PRESET_FAST);
    else if (optflow_method == "DISflow_medium")
        algorithm = createOptFlow_DIS(DISOpticalFlow::PRESET_MEDIUM);
    else
    {
        printf("Wrong optical flow method!\n");
        return -1;
    }

    algorithm->calc(im1, im2, flow);


    Mat flow_image = flowToDisplay(flow);
    namedWindow( "Computed flow", WINDOW_NORMAL );
    imshow( "Computed flow", flow_image );
    waitKey(0);

    return dist;
}





int main(int argc, char* argv[])
{
    string input("c:/p/opencv/samples/data/Megamind.avi");

    VideoCapture cap(input); // open the video file
    if(!cap.isOpened())  // check if we succeeded
        return -1;


    int numFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);
    vector<Mat> frames;
    frames.reserve(numFrame);
    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from video
        if (frame.empty())
            break;
        frames.push_back(frame);
    }

    computeOptFlow(frames[0], frames[1]);
    cout<<"OK"<<endl;
    computeOptFlow(frames[0], frames[2]);
    cout<<"OK"<<endl;

    return 0;
}

/*

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/tracking.hpp>

using namespace cv::ml;
using namespace cv;
using namespace std;

Mat flowToDisplay(const Mat &flow)
{
    Mat flow_split[2];
    Mat magnitude, angle;
    Mat hsv_split[3], hsv, rgb;
    split(flow, flow_split);
    cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);
    normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);
    hsv_split[0] = angle; // already in degrees - no normalization needed
    hsv_split[1] = Mat::ones(angle.size(), angle.type());
    hsv_split[2] = magnitude;
    merge(hsv_split, 3, hsv);
    cvtColor(hsv, rgb, COLOR_HSV2BGR);
    return rgb;
}


int main()
{
    VideoCapture cap(0); // open the video file
    if(!cap.isOpened())  // check if we succeeded
        return 0;

    Mat prev;
    Ptr<DenseOpticalFlow> pof = optflow::createOptFlow_DIS();
    for(;;)
    {
        Mat frame,gray,flo;
        cap >> frame; // get a new frame from camera
        cvtColor(frame,gray,COLOR_BGR2GRAY);
        if (!prev.empty()) {
            pof->calc(prev, gray, flo); // no problem here
            Mat fd = flowToDisplay(flo);
            imshow("f1",fd);
        }
        imshow("F",gray);
        cv::swap(gray,prev);
        int k = waitKey(10);
        if (k == 27) break;
        if (k==' ') pof->collectGarbage();
    }
    return 0;
}
*/
