#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include<stdio.h>
#include<stdlib.h>

int lowerH = 0;
int upperH = 191;
int lowerS = 149;
int upperS = 256;
int lowerV = 0;
int upperV = 200;
CvMat src, dst;
int morph_operatar = 0;
int morph_size = 0;
int morph_elem = 0;
///////////////////////////////////////////////////////////////////////////////////////////////////
void setwindowSettings()
{
    cvNamedWindow("Detect");
    cvCreateTrackbar("LowerH", "Detect", &lowerH, 256, NULL);
    cvCreateTrackbar("UpperH", "Detect", &upperH, 256, NULL);
    cvCreateTrackbar("LowerS", "Detect", &lowerS, 256, NULL);
    cvCreateTrackbar("UpperS", "Detect", &upperS, 256, NULL);
    cvCreateTrackbar("LowerV", "Detect", &lowerV, 256, NULL);
    cvCreateTrackbar("UpperV", "Detect", &upperV, 256, NULL);
}
using namespace cv;
int main(int argc, char* argv[])
{
    //////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////-Setup OpenCV variables and structures/////////////////////////
    CvSize size640x480 = cvSize(640, 480);
    CvCapture* p_capWebcam;
    IplImage* p_imgOriginal;
    IplImage* p_imgProcessed;
    IplImage* p_imgHSV;
    IplImage* p_imgErode;
    CvMemStorage* p_strStorage;
    CvSeq* p_seqCircles;
    char charCheckForEscKey;            // char for checking key press (Esc exits program)
    ////////////////////-read and detect camera-/////////////////////////////////////
    p_capWebcam = cvCaptureFromCAM(0);  // 0 => use 1st webcam, may have to change to a different number if you have multiple cameras
    //cvWaitKey(1);
    if (p_capWebcam == NULL) {          // if capture was not successful . . .
        printf("error: capture is NULL \n");    // error message to standard out . . .
        getchar();                              // getchar() to pause for user see message . . .
        return(-1);                             // exit program
    }
    cvNamedWindow("Original", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Processed", CV_WINDOW_AUTOSIZE);
    p_imgProcessed = cvCreateImage(size640x480, IPL_DEPTH_8U, 1);
    p_imgHSV = cvCreateImage(size640x480, IPL_DEPTH_8U, 3);
    p_imgErode = cvCreateImage(size640x480, IPL_DEPTH_8U, 1);
    setwindowSettings();
    // Main program loop
    while (1)
    {
                p_imgOriginal = cvQueryFrame(p_capWebcam);      // get signal frame from webcam
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        if (p_imgOriginal == NULL) {                    // if frame was not captured successfully . .
            printf("error: frame is NULL \n");      // error message to std out
            getchar();
            break;
        }
        cvCvtColor(p_imgOriginal, p_imgHSV, CV_BGR2HSV);
        cvInRangeS(p_imgHSV, cvScalar(lowerH, lowerS, lowerV), cvScalar(upperH, upperS, upperV), p_imgProcessed);           ////////////////////////////////////////////////////////////////////////////////////////////////////
        p_strStorage = cvCreateMemStorage(0);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        cvSmooth(p_imgProcessed, p_imgProcessed, CV_GAUSSIAN, 9, 9);
            cvErode(p_imgProcessed, p_imgErode, NULL, 3);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cvHoughCircles(p_imgProcessed,p_strStorage,CV_HOUGH_GRADIENT,2,p_imgProcessed->height / 4,100,50,10,400);
        p_seqCircles = cvHoughCircles(p_imgErode, p_strStorage, CV_HOUGH_GRADIENT, 2, 1000, 100, 90, 40, 100);
        cvShowImage("Original", p_imgOriginal);
        cvShowImage("Processed", p_imgProcessed);       // image after processing
        cvShowImage("Erode", p_imgErode);
        cvReleaseMemStorage(&p_strStorage);
        charCheckForEscKey = cvWaitKey(10);
        if (charCheckForEscKey == 27) break;
    }
    cvReleaseCapture(&p_capWebcam);                 // release memory as applicable
    cvDestroyWindow("Original");
    cvDestroyWindow("Processed");
    cvDestroyWindow("Erode");
    //cvDestroyAllWindows();
    return(0);}
