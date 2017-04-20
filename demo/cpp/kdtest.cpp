#include <opencv2/opencv.hpp>
using namespace cv;
using namespace cv::ml;
#include <iostream>
using namespace std;

#include "kdtree.hpp"
/** @function main */
int main( int argc, char** argv )
{
	Mat points(5,15,CV_32F);
	randu(points, 0,100);
	KDTree T(points, false);
	const int K = 3, Emax = INT_MAX;
	Mat idx,neigh,labl;
	float dist[K];
	Mat query_vec(1,15,CV_32F);
	randu(query_vec, 0,100);
	T.findNearest(query_vec, K, Emax, idx, neigh, labl);
	cerr << idx << endl;
	cerr << neigh << endl;
	cerr << labl << endl;
	return(0);
}

