#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;



int main(int argc, char **argv)
{
	Mat m1(10,10,CV_8UC3); m1=17;
	Mat m2(6,8,CV_32F); m2=3;

    FileStorage fs("my.yml", FileStorage::WRITE);

    fs << "object1" << "{";
    fs << "Date of storage" << "Wed Jul 26 16:45:46 2017\n";
    fs << "Mat" << m1;
    fs << "}";

    fs << "object2" << "{";
    fs << "Date of storage" << "Wed Jul 26 16:48:16 2017\n";
    fs << "Mat" << m2;
    fs << "}";

    fs.release();

    fs.open("my.yml", FileStorage::READ);
    FileNode fn = fs["object2"];
    Mat M2;
    fn["Mat"] >> M2;
    String date;
    fn["Date of storage"] >> date;
    cerr << date << endl;
    cerr << M2 << endl;
}
