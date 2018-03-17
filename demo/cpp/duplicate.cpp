#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/img_hash.hpp>
#include "profile.h"

using namespace cv;
using namespace std;


size_t match_bf(const vector<Mat> &keys, Ptr<img_hash::PHash> hash, vector<Point> &dups, double thresh=8) {
	double md = 0; int k=0;
	for (size_t i=0; i<keys.size(); i++) {
		for (size_t j=i+1; j<keys.size(); j++) {
			double d = hash->compare(keys[i],keys[j]);
			if (d<thresh) {
				dups.push_back(Point(i,j));
				cout << i << " " << j << " " << d << endl;
			}
			md += d;
			k++;
		}
	}
	cout << "avg dist " << (md / k) << endl;
	return dups.size();
};

//bool operator == (int a, int b) {return true;}
int main(int argc, char** argv)
{
	String folder = "c:/data/img/cache/15/*.jpg"; // 17
	vector<String> fn;
	glob(folder,fn);

	Ptr<img_hash::PHash> hash = img_hash::PHash::create();
	vector<Mat> keys;
	for (auto f:fn) {
		Mat img, key;
		{
			PROFILEX("read")
			img = imread(f);
		}
		{
			PROFILEX("hash")
			hash->compute(img,key);
			keys.push_back(key);
		}
	}
	cout << keys.size() << endl;
	vector<Point> dups;
	{
		PROFILEX("bf_match")
		match_bf(keys,hash,dups,6);
	}
	cout << dups.size() << endl;
	for (size_t i=0; i<dups.size(); i++) {
		Mat a = imread(fn[dups[i].x]);
		Mat b = imread(fn[dups[i].y]);
		cout << keys[dups[i].x] << endl;
		cout << keys[dups[i].y] << endl << endl;
		imshow("A",a);
		imshow("B",b);
		waitKey();
	}
    return 0;
}
