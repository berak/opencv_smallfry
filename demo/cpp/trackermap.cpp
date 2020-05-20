#include "opencv2/opencv.hpp"
#include "opencv2/tracking.hpp"
using namespace cv;

#include <iostream>
using namespace std;

typedef cv::Ptr<cv::Tracker> (*TrackerFactory)();

template<class T>
cv::Ptr<cv::Tracker> _wrap(){ return T::create(); }

int main()
{
	std::map<std::string, TrackerFactory> trackerTypes_map;
	trackerTypes_map["BOOSTING"] = _wrap<cv::TrackerBoosting>;
	trackerTypes_map["MIL"]      = _wrap<cv::TrackerMIL>;
	trackerTypes_map["MOSSE"]    = _wrap<cv::TrackerMOSSE>;

	auto test = [&](string s){
		cout << s << "\t\t";
		TrackerFactory factory = trackerTypes_map[s]; // returns nullptr for unknown keys
		cout << factory;
		if (factory) {
			Ptr<Tracker> p1 = factory();
			cout << " " << (!p1.empty());
		}
		cout << endl;
	};
	test("MOSSE");
	test("MIL");
	test("Boostings");
	test("BOOSTING");

	return 0;
}
