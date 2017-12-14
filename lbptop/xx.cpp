#include <iostream>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

void w() {
	FileStorage fs("my.yml", 1);
	Mat m(3,3,CV_32F,17.0f);
	Mat l(1,3,CV_32S,3);
	fs << "sequences" << "[";
	for (int i=0; i<4; i++) {
		fs << "{";
		fs << "points" << "[";
		for (int j=0; j<3; j++) {
			fs << m;
		}
		fs << "]";
		fs << "labels" << l;
		fs << "}";
	}
	fs << "]";
}

int main(int argc, char **argv) {
	w();
    vector< vector<Mat> > data;
    vector< vector<int> > labels;
    FileStorage fs("my.yml",0);

    FileNode n = fs["sequences"];
    FileNodeIterator si = n.begin();
    for (; si != n.end(); ++si) {
    	FileNode p = (*si)["points"];
	    FileNodeIterator pi = p.begin();
	    for (; pi != p.end(); ++pi) {
	    	Mat m;
	    	(*pi) >> m;
	    	cout << m << endl;
	    }
    	Mat l;
    	(*si)["labels"] >> l;
    	cout << l << endl;
    }


    return 0;
}
