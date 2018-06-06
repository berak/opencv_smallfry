#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/aruco.hpp"
using namespace std;
using namespace cv;
using namespace cv::ml;

/**
 */
static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return true;
}


int main(int argc, char** argv)
{
    namedWindow("G");
    int brate=80;
    int otsu=20;
    int erro=20;
    int adap=20;
    createTrackbar("1","G",&brate,100);
    createTrackbar("O","G",&otsu,100);
    createTrackbar("E","G",&erro,100);
    createTrackbar("A","G",&adap,100);
    Mat img = imread("markers.png");
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    readDetectorParameters("detector_params.yml", parameters);

/*   for(parameters->maxErroneousBitsInBorderRate=0.5;parameters->maxErroneousBitsInBorderRate<1.0;parameters->maxErroneousBitsInBorderRate+=0.2)
   for(parameters->minOtsuStdDev=0.1; parameters->minOtsuStdDev<10.0; parameters->minOtsuStdDev+=0.1)
   for(parameters->adaptiveThreshConstant=0.5; parameters->adaptiveThreshConstant<20.0; parameters->adaptiveThreshConstant+=0.2)
   for(parameters->errorCorrectionRate=0.02; parameters->errorCorrectionRate<0.1; parameters->errorCorrectionRate+=0.01)
  // for(parameters->minMarkerDistanceRate=1; parameters->minMarkerDistanceRate<20.0; parameters->minMarkerDistanceRate+=1)
*/
//   while (1) {
    Mat grey = img.clone();
   //  parameters->polygonalApproxAccuracyRate = double(brate)/100;
    // parameters->perspectiveRemovePixelPerCell = double(otsu)/10;
    // parameters->minMarkerPerimeterRate = double(erro)/100;
    // parameters->adaptiveThreshWinSizeStep = 0.01+double(adap)/5;
    //bool ok = readDetectorParameters("detector_params.yml", parameters);
    cout << parameters->polygonalApproxAccuracyRate << " " <<  parameters->errorCorrectionRate <<" " <<  parameters->adaptiveThreshConstant <<" " <<  parameters->minOtsuStdDev <<" " <<  parameters->maxErroneousBitsInBorderRate << "\t";
    cv::aruco::detectMarkers(grey, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
    cout << markerCorners.size() << " " << markerIds.size() << " " << rejectedCandidates.size() << endl;
    aruco::drawDetectedMarkers(grey, rejectedCandidates, noArray(), Scalar(100, 0, 255));
    aruco::drawDetectedMarkers(grey, markerCorners, noArray(), Scalar(100, 255));
    imshow("G", grey);
    int k = waitKey();
    ///if (k==27) break;
//}

    return 0;
}
