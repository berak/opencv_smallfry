#include <opencv2/opencv.hpp>
#include "matching.h"
#include "profile.h"

using namespace cv;
using namespace std;


int main(int argc, char **argv) {
	// cmdline args
	String match = argc>1 ? argv[1] : "onedollar";
	String scene = argc>2 ? argv[2] : "scene1.png";
	int sizeThresh = argc>3 ? atoi(argv[3]) : 200;

	// "object factory"
	matching::Distance *distance;
	Ptr<matching::Matcher> matcher;
	if (match == "hausdorff") { matcher = hausdorff::createMatcher(); distance = &hausdorff::distance; }
	if (match == "onedollar") { matcher = onedollar::createMatcher(); distance = &onedollar::distance; }
	if (match == "fourier")   { matcher = fourier::createMatcher();   distance = &fourier::distance;   }
	// broken (needs resampling of the shapes to equal size)
	//if (match == "sampson")   { matcher = sampson::createMatcher();   distance = &sampson::distance;   }

	// img processing
	Mat m1, m2, m = imread(scene, IMREAD_GRAYSCALE);
	//Mat mc; cvtColor(m/8, mc, COLOR_GRAY2BGR); // for drawing
	bitwise_not(m, m1);
	blur(m1,m1,Size(6,6));
	threshold(m1, m2, 5, 255, THRESH_BINARY);

	// contours, filtered by size
	vector<vector<Point> > raw_contours, contours;
	findContours(m2, raw_contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	for (int i=0; i<raw_contours.size(); i++)
	{
	    if (raw_contours[i].size() >= sizeThresh)
	    {
	        contours.push_back(raw_contours[i]);
	        //drawContours(mc, raw_contours, i, Scalar(128,128,128), 2);
	        {
	        	PROFILEX("matcher.add");
	        	matcher->add(raw_contours[i]);
	    	}
	    }
	}
	cerr << contours.size() << " filtered contours." << endl;

	// NxN distance matrix
	size_t N = contours.size();
	Mat_<double> confusion(N,N,0.0);
	for (size_t i=0; i<N; i++) {
		for (size_t j=0; j<N; j++) {
			PROFILEX("distance");
			confusion(i, j) = distance(contours[i], contours[j]);
		}
    }
    cerr << match << endl;
    cerr << confusion << endl;

    Mat_<int> closest_i(1,N);
    Mat_<double> closest_d(1,N);
    for (size_t i=0; i<N; i++) {
    	PROFILEX("matcher.match");
    	vector<Point2d> best;
    	int id;
    	double d;
    	matcher->match(contours[i],best,d,id);
    	closest_i(0,i) = id;
    	closest_d(0,i) = d;
	}
	cerr << closest_i << endl;
	cerr << closest_d << endl;

    /*
	int  ca=3, cb=2;
	float alpha, phi, s;
	double d = distance(contours[ca], contours[cb], alpha, phi, s);
    cout <<"Self Distance for contour " << ca << " and " << ca << " is " << distance(contours[ca], contours[ca]) << endl;
    cout <<"Self Distance for contour " << cb << " and " << cb << " is " << distance(contours[cb], contours[cb]) << endl;
    drawContours(mc, contours, ca, Scalar(0,0,255), 2);
    drawContours(mc, contours, cb, Scalar(0,255,0), 2);
    cout <<"Distance between contour " << ca << " and " << cb << " is " << d << " " << endl;
	// reconstruct (fourier only)
	//vector<Point> rec = fourier::reconstruct(contours[cb],alpha,-phi,s); // a from b
	vector<Point> rec = fourier::reconstruct(contours[ca], alpha, phi, s); // b from a
    int last = contours.size();
    contours.push_back(rec);
    drawContours(mc, contours, last, Scalar(255,0,0), 2, LINE_AA, noArray(), INT_MAX, Point(120,120));
	imshow("mc ",mc);
	waitKey();
    */
    return 0;
}
