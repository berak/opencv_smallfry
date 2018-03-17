#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>

using namespace cv;
using namespace std;

std::vector<String> readClassNames(const char *filename)
{
    std::vector<String> classNames;

    std::ifstream fp(filename);
    if (!fp.is_open())
    {
        std::cerr << "File with classes labels not found: " << filename << std::endl;
        exit(-1);
    }

    std::string name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        if (name.length())
            classNames.push_back( name );
    }

    fp.close();
    return classNames;
}


int main(int argc, char** argv)
{
	theRNG().state = getTickCount();

	Mat img0 = imread("c:/data/caltech/101_ObjectCategories/beaver/image_0004.jpg");
    Mat img1 = imread("c:/data/caltech/101_ObjectCategories/beaver/image_0005.jpg");
    Mat img2 = imread("c:/data/caltech/101_ObjectCategories/ant/image_0002.jpg");
	cout << img0.size() << img1.size() << img2.size() << endl;

	vector<String> cls = readClassNames("c:/data/mdl/imagenet_comp_graph_label_strings.txt");
    dnn::Net net = dnn::readNetFromTensorflow("c:/data/mdl/tensorflow_inception_graph.pb");
    auto proc = [&](Mat &image) {
		Mat inputBlob = dnn::blobFromImage(image, 1.0f, Size(224, 224), Scalar(), true, false);
		net.setInput(inputBlob);
		//return net.forward("avgpool0").reshape(1,1).clone();
		return net.forward("nn0/reshape").reshape(1,1).clone();
	};
    auto predict = [&](Mat &image) {
		Mat inputBlob = dnn::blobFromImage(image, 1.0f, Size(224, 224), Scalar(), true, false);
		net.setInput(inputBlob);
		Mat pred = net.forward().reshape(1,1);
	    Point classNumber; double prob;
	    minMaxLoc(pred, NULL, &prob, NULL, &classNumber);
	    cout << "predicted " << classNumber.x << " " << prob << " " << cls[classNumber.x] << endl;;
		return classNumber.x;
	};
	imshow("I0",img0);
	imshow("I1",img1);
	imshow("I2",img2);
	resize(img1,img1,img2.size());
	Mat f0 = proc(img0);
	Mat f1 = proc(img1);
	Mat f2 = proc(img2);
	double d0 = norm(f0,f1);
	double d1 = norm(f1,f2);
	cout << "d0 " << d0 << endl;
	cout << "d1 " << d1 << endl;
	int c0 = predict(img0);
	int c2 = predict(img2);
	Mat N(img2.size(), CV_32FC3,Scalar::all(0));
	Mat n(img2.size(), CV_32FC3);
	Mat img3,im2f; img2.convertTo(im2f, CV_32F);
	double db=d1;
	for (int i=0; i<10000; i++) {
		randn(n,0,2.5);
		img3 = im2f + N + n;
		img3.convertTo(img3, CV_8U);
		Mat f3 = proc(img3);
		//Mat loss = f1 - f3;
		//Mat u = n.reshape(1,1);
		double d = norm(f1,f3);
		cout << i << " " << d << "                  \r";
		if (d<db) {
			db=d;
			N+=n;
			double ds = sum(abs(n+N))[0]; // noise
			double dg = norm(img1,img3); // bgr distance
			cout << i << " " << d << " " << ds << " " << dg << " ";
			int c3 = predict(img3);
			imshow("I",img3);
			waitKey(20);
			//if (d < d0) break;
			if (c3 == c0) break;
		}
	}

	normalize(N,N,0,1,NORM_MINMAX);
	imshow("N",N);
	waitKey();
    return 0;
}

/*
layer:avgpool0, noise:2.5
d0 104.695
d1 113.727
predicted 195 0.506143 beaver
predicted 224 0.862497 ant
28 113.584 237968 47728.8 predicted 224 0.861109 ant
...
130 107.612 575375 47579.8 predicted 608 0.317425 tarantula
...
769 93.9911 1.40177e+006 47563.3 predicted 48 0.298935 weasel
...
807 93.1353 1.43822e+006 47557.7 predicted 608 0.281455 tarantula
...
808 93.0473 1.44454e+006 47559.6 predicted 48 0.296634 weasel
...
3240 67.0075 2.36113e+006 47714.4 predicted 23 0.447 mink
...
4877 57.4204 2.65721e+006 47821.4 predicted 195 0.424999 beaver
*/
