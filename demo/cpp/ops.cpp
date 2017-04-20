#include "opencv2/opencv.hpp"
using namespace cv;

template<int T> int op   () {
//#error not implementd.
 return -1;
}

template<>      int op<1>() { return 1; }
template<>      int op<2>() { return 4; }


int main() {
	int n =  op<2>();
	std::cout << n;
	return 0;
}
