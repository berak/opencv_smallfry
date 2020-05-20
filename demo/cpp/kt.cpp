#include "opencv2/opencv.hpp"
#include "kuhn_munkres.hpp"
#include <iostream>
using namespace cv;
using namespace std;

int main() {
	// https://brilliant.org/wiki/hungarian-matching/
	Mat_<float> dis(3,4);
	dis << 108,125,150,122,
		   150,135,175,11,
		   122,148,250;213;

	KuhnMunkres k;
	vector<size_t> res = k.Solve(dis);

	for (auto v:res)
		cout << v << endl;
	return 0;
}