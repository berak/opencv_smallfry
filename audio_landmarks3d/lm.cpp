#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace std;

#include "AudioFile.h"

int main(int argc, char const *argv[])
{
	int sr = 8000; // assumes: 8khz mono
	int num_frames = 7;
	int increment  = sr * 0.04; // 25 fps
	int sample_len = num_frames * increment;
	int sample_pos = 0;

    dnn::Net nn = dnn::readNet("model.onnx");

    FileStorage fs("lm.yml", 0);
    Mat mean_shape, eigen_vectors;
    fs["mean_shape"] >> mean_shape;
    fs["eigen_vectors"] >> eigen_vectors;
    cout << mean_shape.size() << eigen_vectors.size() << endl;

	AudioFile<float> audio;
	audio.load("S2.WAV");
	audio.printSummary();
	vector<float> buf = audio.samples[0];
	double m,M;
	minMaxLoc(cv::abs(Mat(buf)), &m, &M, 0, 0);
	Mat(buf) /= M;
	cout << ((buf.size()-sample_len) / increment) << " frames." << endl;

	while (sample_pos < buf.size() - sample_len) {
		int sz[]{1, 1, sample_len};
		Mat in(3, sz, CV_32F, buf.data() + sample_pos);
		sample_pos += increment;

	    nn.setInput(in);
	    Mat res = nn.forward();
	    res = res.t();
	    Mat_<float> pts = mean_shape.clone();
	    for (int i=0; i<eigen_vectors.cols; i++) {
	    	pts(i) += res.dot(eigen_vectors.col(i));
	    }
	    pts = pts.reshape(1,68);

	    int W=400, H=400;
	    Mat draw(H, W, CV_8UC3, Scalar::all(128));
	    for (int i=0; i<pts.rows; i++) {
	    	int x = pts(i,0) * W*2 + W/2;
	    	int y = pts(i,1) * H*2 + H/2;
	    	circle(draw, Point(x,y), 3, Scalar(50,50,255), -1);
	    }
	    imshow("draw", draw);
	    waitKey(60);
	}
	waitKey();
    return 0;
}