/************************************************************************
* File:	CompressiveTracker.h
* Brief: C++ demo for paper: Kaihua Zhang, Lei Zhang, Ming-Hsuan Yang,"Real-Time Compressive Tracking," ECCV 2012.
* Version: 1.0
* Author: Yang Xian
* Email: yang_xian521@163.com
* Date:	2012/08/03
* History:
* Revised by Kaihua Zhang on 14/8/2012, 23/8/2012
* Email: zhkhua@gmail.com
* Homepage: http://www4.comp.polyu.edu.hk/~cskhzhang/
* Project Website: http://www4.comp.polyu.edu.hk/~cslzhang/CT/CT.htm
************************************************************************/
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#ifndef MAX_PATH
#define MAX_PATH 4096
#endif

using std::vector;
using namespace cv;
//---------------------------------------------------
class CompressiveTracker
{
public:
    CompressiveTracker(void);
	~CompressiveTracker(void);

private:
	int featureMinNumRect;
	int featureMaxNumRect;
	int featureNum;
	vector<vector<Rect> > features;
	vector<vector<float> > featuresWeight;
	int rOuterPositive;
	vector<Rect> samplePositiveBox;
	vector<Rect> sampleNegativeBox;
	int rSearchWindow;
	Mat imageIntegral;
	Mat samplePositiveFeatureValue;
	Mat sampleNegativeFeatureValue;
	vector<float> muPositive;
	vector<float> sigmaPositive;
	vector<float> muNegative;
	vector<float> sigmaNegative;
	float learnRate;
	vector<Rect> detectBox;
	Mat detectFeatureValue;
	RNG rng;

private:
	void HaarFeature(const Rect2d& _objectBox, int _numFeature);
	void sampleRect(const Mat& _image, const Rect2d& _objectBox, float _rInner, float _rOuter, int _maxSampleNum, vector<Rect>& _sampleBox);
	void sampleRect(const Mat& _image, const Rect2d& _objectBox, float _srw, vector<Rect>& _sampleBox);
	void getFeatureValue(Mat& _imageIntegral, vector<Rect>& _sampleBox, Mat& _sampleFeatureValue);
	void classifierUpdate(Mat& _sampleFeatureValue, vector<float>& _mu, vector<float>& _sigma, float _learnRate);
	void radioClassifier(vector<float>& _muPos, vector<float>& _sigmaPos, vector<float>& _muNeg, vector<float>& _sigmaNeg,
						Mat& _sampleFeatureValue, float& _radioMax, int& _radioMaxIndex);
public:
	void processFrame(const Mat& _frame, Rect2d& _objectBox);
	void init(const Mat& _frame, const Rect2d& _objectBox);
};
