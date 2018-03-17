#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/img_hash.hpp>
#include <fstream>

using namespace cv;

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	ifstream is("04.bmp", std::ios::binary);
	is.seekg (0, ios::end);
	size_t length = is.tellg();
	cout << length << endl;
	is.seekg (0, ios::beg);
	char *data = new char[length];
	is.read(data,length);

	//vector<uchar> bytes(data,data+length);
	Mat bytes(1,length,CV_8S,data);
	Mat m = imdecode(bytes,-1);
	imshow("M",m);
	waitKey();
    return 0;
}
