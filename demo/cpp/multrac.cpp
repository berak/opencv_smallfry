#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
using namespace cv;
using namespace std;


int main()
{
    Ptr<MultiTracker> trackers; // initially empty !

	cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    //hog.winSize = Size(48,96);
	//hog.setSVMDetector(cv::HOGDescriptor::getDaimlerPeopleDetector());

	VideoCapture cap("../img/768x576.avi");

	for (;;) {
		Mat img;
	    if (cap.grab() == false) {
	        std::cout << "movie's over." << std::endl;
	        break;
	    }
        cap.retrieve(img);
        //resize(img, img, Size(640, 480));

        if (trackers.empty()) {
        	// 1. hog detection:
	        vector<Rect> found;
	        hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
			cerr << "found     " << found.size() << endl;
			// 2. copy to rect2d:
	        vector<Rect2d> v2;
			for (size_t i=0; i<found.size(); i++) {
                v2.push_back(found[i]);
        	}
	        // 3. (re)initialize trackers:
	        trackers = makePtr<MultiTracker>("MIL");
	        trackers->add(img, v2);
        }


        if (trackers->update(img)) {
	        for (size_t i = 0; i<trackers->objects.size(); i++) {
	            rectangle(img, trackers->objects[i], Scalar(255, 0, 0), 2, 1);
	        }
        } else {
        	trackers.release();
        }

        imshow("trac", img);
        int key = waitKey(40);
    	if (key == 27)
            break;
    	if (key == ' ')
            trackers.release(); // start new hog detection & reinit trackers
    }

	return 0;
}
