#include "opencv2/opencv.hpp"
#include <iostream>
#include "profile.h"
#include "lbptop.h"

using namespace cv;
using namespace std;

// -----8<-----------------------------------------------------------
static const vector<Point> samps_face { // 15 fixed, tl() landmarks in (96,96) space
        	      Point(22,8),  Point(50,8),
	Point(12,24), Point(24,20), Point(48,20), Point(60,24),
    Point(16,40), Point(22,40), Point(50,40), Point(54,40),
        	  Point(16,60), Point(36,60), Point(54,60),
        	      Point(22,70), Point(50,70)
};

// -----8<-----------------------------------------------------------
int make_data() {
    int BS = 24;
    lbp_set_blocksize(BS); // only 1 single block in t for whole video, 8496 features.

	CascadeClassifier cad("haarcascade_frontalface_alt.xml");
	String path = "c:/data/faces/ckplus/";
	vector<String> emos;
	glob(path+"Emotion/*.txt", emos, true);
	cerr << emos.size() << endl;
	vector<float> hist(8,0);
	Mat data,labels;
	for (auto f:emos) {
		ifstream emo(f);
		float e;
		emo >> e;
		hist[(int)e] ++;
		int l = f.find_last_of('\\');
		String txt = f.substr(l + 1);
		String sub = txt.substr(0, 4);
		String em  = txt.substr(5, 3);
		String frm = txt.substr(9, 8);
		cerr << e << " " << sub << " " << em << " " << frm << endl;
		String imgbase = path + "cohn-kanade-images/" + sub + "/" + em + "/";
		int nfrm = stoi(frm);
		deque<Mat> images;
		Rect box;
		for (int i=0; i<nfrm; i++) {
			Mat gray;
			{ PROFILEX("load")
			gray = imread(imgbase+txt.substr(0,17)+".png", 0);
			if (gray.empty())
				continue;
			}
			equalizeHist(gray,gray);
	    	if (i==0) {
	    		PROFILEX("detect")
				vector<Rect> faces;
		   		cad.detectMultiScale(gray,faces,1.1,4,CV_HAAR_FIND_BIGGEST_OBJECT,cv::Size(BS,BS));
			    if (faces.size()) {
			    	box = faces[0];
			    }
	    	}
	    	Mat det;
	    	resize(gray(box), det, Size(96,96));
			images.push_back(det);
			rectangle(gray, box, Scalar(200), 1);
			imshow("F",gray);
			waitKey(1);
		}
    	// procrustes
		while(images.size() < BS) {
			int N = int(images.size()) - 1;
			int n = theRNG().uniform(0,N);
			images.insert(images.begin() + n, images.at(n));
		}
		while(images.size() > BS) {
			int N = int(images.size()) - 1;
			int n = theRNG().uniform(0,N);
			images.erase(images.begin() + n);
		}

		Sequence seq;
		for (auto i:images) seq.push_back(i);
		Mat hist = lbptop(seq, samps_face);

		data.push_back(hist);
		labels.push_back((int)e);
	}
	FileStorage fs("ckplus_lbp.yml.gz",1);
	fs << "labels" << labels;
	fs << "data" << data;
	fs.release();
	cerr << Mat(hist).t() <<  endl;
	return 0;
}

int main() {
	//make_data();

	Mat data, labels;
	FileStorage fs("ckplus_lbp.yml.gz",0);
	fs["labels"] >> labels;
	fs["data"] >> data;
	fs.release();

	Ptr<ml::TrainData> tdata = ml::TrainData::create(data,0,labels);
    tdata->setTrainTestSplitRatio(0.7);
    data   = tdata->getTrainSamples();
    labels = tdata->getTrainResponses();
    labels.convertTo(labels, CV_32S); // hrrm!
    cerr << "training with " << data.size() << endl;

    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setKernel(ml::SVM::LINEAR);
    svm->train(data,0,labels);

    Mat vdata = tdata->getTestSamples();
    Mat truth = tdata->getTestResponses();
    cerr << "testing with " << vdata.size() << endl;

    Mat predict;
    svm->predict(vdata, predict);
    predict.convertTo(predict, CV_32S); // hrrm!

    float correct = countNonZero(predict == truth);
    float accuracy = correct / predict.total();
    cerr << "accuracy: " << accuracy << endl;

    // accuracy alone is not enough here, since it might
    //   simply have missed all positives !
    Mat_<int> confusion(8,8,0);
    for (int i=0; i<predict.rows; i++) {
        int p = (int)predict.at<int>(i);
        int t = (int)truth.at<int>(i);
        confusion(p,t) ++;
    }
    cerr << "confusion:\n" << confusion << endl;
    //cerr << truth.t() << endl;
    //cerr << predict.t() << endl;
    return 0;
}
