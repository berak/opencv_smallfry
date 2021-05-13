#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

//
// https://github.com/ZFTurbo/Weighted-Boxes-Fusion
// https://arxiv.org/pdf/1910.13302.pdf
//

template<class _Tp>
struct WeightedBoxesFusion {
	typedef Rect_<_Tp> RECT;

	struct Box {
		int label;
		float x1,y1,x2,y2, score, weight;

		Box() {}
		Box(int label, float x1,float y1,float x2,float y2, float score, float weight)
		: label(label), x1(x1), y1(y1), x2(x2), y2(y2), score(score), weight(weight)
		{
		}

		double distance(const Box &b) const {
			RECT r1(x1,y1,x2-x1,y2-y1);
			RECT r2(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1);
			return jaccardDistance(r1,r2);
		}
	};

	vector<Box> B;
	vector<pair<vector<Box>,Box>> FL; // combined F and L lists
	int num_models = 0;
	double weights = 0;
	double THR_IOU = (1.0 - 0.55); // jaccardDistance returns (1-iou)

	bool addModel(const vector<RECT> &rects, const vector<float> &scores, const vector<int> &labels, float weight, float conf_thresh) {
		CV_Assert(rects.size()==scores.size());
		CV_Assert(scores.size()==labels.size());
		for (size_t i=0; i<rects.size(); i++) {
			if (scores[i] < conf_thresh)
				continue;
			B.push_back(Box(labels[i],
				            rects[i].x, rects[i].y, rects[i].x+rects[i].width, rects[i].y+rects[i].height,
				            scores[i] * weight, weight));
		}
		num_models ++;
		weights += weight;
		return true;
	}

	bool fuse(vector<RECT> &rects, vector<float> &scores, vector<int> &labels) {
		sort(B.begin(), B.end(), [](const Box &a, const Box &b) {
			return a.score > b.score;
		});
		// 3.
		for (const auto b : B) {
			double min_d = THR_IOU;
			int best = -1;
			for (int i=0; i<FL.size(); i++) {
				auto &f = FL[i];
				if (b.label != f.second.label)
					continue;
				double d = b.distance(f.second);
				if (d < min_d) {
					best = i;
					min_d = d;
				}
			}
			if (best != -1) {
				FL[best].first.push_back(b);
			} else {
				FL.push_back(make_pair(vector<Box>{b},b));
			}
		}
		// 6.
		for (auto f : FL) {
			float x1=0, x2=0, y1=0, y2=0, sum_score=0;
			for (auto q : f.first) {
				x1 += q.score * q.x1; y1 += q.score * q.y1;
				x2 += q.score * q.x2; y2 += q.score * q.y2;
				sum_score += q.score;
			}
			x1 /= sum_score;
			y1 /= sum_score;
			x2 /= sum_score;
			y2 /= sum_score;
			rects.push_back(RECT(x1,y1,x2-x1,y2-y1));
			scores.push_back(sum_score / weights);
			labels.push_back(f.second.label);
		}
		return true;
	}
};


// example_wbf_2_models
using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	WeightedBoxesFusion<double> wbf;

    vector<Rect2d> boxes1 {
    	Rect2d(0.00, 0.51, 0.81-0.00, 0.91-0.51),
        Rect2d(0.10, 0.31, 0.71-0.10, 0.61-0.31),
        Rect2d(0.01, 0.32, 0.83-0.01, 0.93-0.32),
        Rect2d(0.02, 0.53, 0.11-0.02, 0.94-0.53),
        Rect2d(0.03, 0.24, 0.12-0.03, 0.35-0.24)};
    vector<Rect2d> boxes2 {
        Rect2d(0.04, 0.56, 0.84-0.04, 0.92-0.56),
        Rect2d(0.12, 0.33, 0.72-0.12, 0.64-0.33),
        Rect2d(0.38, 0.66, 0.79-0.38, 0.95-0.66),
        Rect2d(0.08, 0.49, 0.21-0.08, 0.89-0.49)};

    vector<float> scores1 {0.9,0.8,0.2,0.4,0.7};
    vector<float> scores2 {0.5,0.8,0.7,0.3};

    vector<int> labels1 {0,1,0,1,1};
    vector<int> labels2 {1,1,1,0};

    wbf.addModel(boxes1, scores1, labels1, 2, 0.05);
    wbf.addModel(boxes2, scores2, labels2, 1, 0.05);

    vector<Rect2d> boxes;
    vector<float> scores;
    vector<int> labels;

    wbf.fuse(boxes,scores,labels);

    for (auto b : boxes)
    	cout << b.x << " " << b.y << " " << b.x+b.width << " " << b.y + b.height << endl;
    cout << Mat(scores).t() << endl;
    cout << Mat(labels).t() << endl;

    return 0;
}
/**
1st 2 entries are switched wrt the python version, i don't sort for labels here

0.00181818 0.475455 0.813636 0.913636
0.106667 0.316667 0.713333 0.62
0.03 0.24 0.12 0.35
0.02 0.53 0.11 0.94
0.38 0.66 0.79 0.95
0.04 0.56 0.84 0.92
0.08 0.49 0.21 0.89
[0.73333335, 0.80000001, 0.46666667, 0.26666668, 0.23333333, 0.16666667, 0.1]
[0, 1, 1, 1, 1, 1, 0]
**/