#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
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

	WeightedBoxesFusion(double iou_thresh=0.45) : THR_IOU(iou_thresh) {}

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
		// 3.1
        sort(B.begin(), B.end(), [](const Box &a, const Box &b) {
			return a.score > b.score;
		});
		// 3.3
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
				FL[best].first.push_back(b); // 3.5
			} else {
				FL.push_back(make_pair(vector<Box>{b},b)); // 3.4
			}
		}
		// 3.6
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
			scores.push_back(sum_score / weights); # see https://github.com/ZFTurbo/Weighted-Boxes-Fusion/pull/25
			labels.push_back(f.second.label);
		}
		return true;
	}
};

int test_wbf_2_models() {
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
    /** output:
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
}

#include <map>
using namespace std;
using namespace cv;

struct Boxes {
    string name;
    vector<Rect2d> rects;
    vector<float> conf;
    vector<int> label;
    void print() {
        cout << name << endl;
        for (size_t i=0; i<rects.size(); i++)
            cout << label[i] << " " << conf[i] << " " << rects[i] << endl;
    }
    void draw(Mat &img, Scalar color) {
        for (size_t i=0; i<rects.size(); i++) {
            Rect rec(rects[i].x * img.cols, rects[i].y * img.rows,
                     rects[i].width * img.cols, rects[i].height * img.rows);
            rectangle(img, rec, color * conf[i], 1);
        }
    }
};

// filename
// num_rects
// x y w h conf label
int read_strm(string strm, map<string,Boxes> &bx) {
    ifstream f1(strm.c_str());
    while (f1.good()) {
        Boxes b;
        f1 >> b.name;
        if (b.name.empty())
            break;
        int n;
        f1 >> n;
        for (int i=0; i<n; i++) {
            Rect2d r;
            float c;
            int id;
            if (f1 >> r.x >> r.y >> r.width >> r.height >> c >> id) {
                b.rects.push_back(r);
                b.conf.push_back(c);
                b.label.push_back(id);
            }
        }
        bx.insert(make_pair(b.name,b));
    }
    return 0;
}

// fuse 2 preserialized models
int ensemble() {
    namedWindow("Fusion",0); // resizable
    int shared=0;
    map<string,Boxes> bx,by;
    read_strm("nano.txt",bx);
    cout << "nano " << bx.size() << endl;
    read_strm("yolo4.txt",by);
    cout << "yolo " << by.size() << endl;
    for (auto b : bx) {
        auto it = by.find(b.first);
        if (it != by.end()) {
            shared ++;

            WeightedBoxesFusion<double> wbf(0.45);
            Boxes &box1 = b.second;
            Boxes &box2 = (*it).second;
            wbf.addModel(box1.rects, box1.conf, box1.label, 2.5, 0.02);
            wbf.addModel(box2.rects, box2.conf, box2.label, 2.5, 0.02);

            Boxes res; res.name=b.first;
            wbf.fuse(res.rects, res.conf, res.label);
            cout << box1.rects.size() << " " << box2.rects.size() << " " << res.rects.size() << endl;
            res.print();

            Mat img = imread(b.first);
            CV_Assert(!img.empty());
            box1.draw(img, Scalar(100,40,40));
            box2.draw(img, Scalar(40,40,100));
            res.draw(img, Scalar(0,200,0));
            imshow("Fusion",img);
            waitKey();
        }
    }
    cout << "shar " << shared << endl;
    return 0;
}

//
// generate a file with detections, in the format:
//
// filename
// num_rects
// x y w h conf label
//
int yolo(string tn, bool do_flip=false) {
    string model = "c:/data/dnn/yolo/yolov3.weights";
    string cfg =  "c:/data/dnn/yolo/yolov3.cfg";
    if (tn == "nano") {
        model = "c:/data/dnn/yolo/MobileNetV2-YOLOv3-Nano-coco.weights";
        cfg = "c:/data/dnn/yolo/MobileNetV2-YOLOv3-Nano-coco.cfg";
    } else
    if (tn == "yolo4") {
        model = "c:/data/dnn/yolo/yolov4-tiny.weights";
        cfg = "c:/data/dnn/yolo/yolov4-tiny.cfg";
    }
    std::ofstream txt(tn + (do_flip ? "_flip.txt" : ".txt"));

    float confThreshold = 0.6;
    float nmsThreshold = 0.4;

    vector<string> classes;
    std::string file = "c:\\p\\opencv\\samples\\data\\dnn\\object_detection_classes_yolov3.txt";
    std::ifstream ifs(file.c_str());
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }
    cout << classes.size() << " classes." << endl;

    dnn::Net net = dnn::readNet(model,cfg);
    std::vector<String> outNames = net.getUnconnectedOutLayersNames();
    for (string s:outNames) {
        cout << s << endl;
    }

    String c_in("c:\\data\\cache\\24\\*.png");
    vector<String> fn;
    glob(c_in, fn, true);
    cout << fn.size() << " files." << endl;
    for(size_t i=0; i<fn.size(); i++)
    {
        cout << fn[i] << endl;
        Mat im = imread(fn[i]);
        CV_Assert(! im.empty());
        if (do_flip)
            flip(im,im,1);
        //Mat blob = dnn::blobFromImage(im, 0.00392, Size(416,416), Scalar(0), true, false);
        Mat blob = dnn::blobFromImage(im, 0.00392, Size(512,512), Scalar(0), true, false);
        net.setInput(blob);
        vector<Mat> res;
        net.forward(res, outNames);

        vector<Rect> rects;
        vector<float> confs;
        vector<int> ids;
        for (int j=0; j<res.size(); j++)
        {
            Mat level = res[j];
            for (int k=0; k<level.rows; k++)
            {
                Mat_<float> row = level.row(k);
                Mat scores = row.colRange(5,level.cols);
                double v;
                Point p;
                minMaxLoc(scores,0,&v,0,&p);
                if (v>confThreshold)
                {
                    float cx = row(0) * im.cols;
                    float cy = row(1) * im.rows;
                    float w  = row(2) * im.cols;
                    float h  = row(3) * im.rows;
                    rects.push_back(Rect(cx-w/2, cy-h/2, w, h));
                    confs.push_back(v);
                    ids.push_back(p.x);
                }
            }
        }
        if (rects.size()) {
            cout << "rects " << rects.size() << endl;
            txt << fn[i] << endl;
            txt << rects.size() << endl;
            for (size_t k=0; k<rects.size(); k++) {
                float x1 = rects[k].x;
                if (do_flip) {
                    x1 = im.cols - rects[k].x - rects[i].width;
                }
                txt << (float(x1)/im.cols) << " " << (float(rects[k].y)/im.rows) << " " << (float(rects[k].width)/im.cols) << " " << (float(rects[k].height)/im.rows) << " " << confs[k] << " " << ids[k] << endl;
            }
        }
        Rect bounds(0,0,im.cols,im.rows);
        vector<int> indices;
        dnn::NMSBoxes(rects, confs, confThreshold, nmsThreshold, indices);
        for (size_t i = 0; i < indices.size(); i++)
        {
            size_t idx = indices[i];
            rectangle(im,rects[idx],Scalar(255,0,0),2);
            cout << idx << "\t" << classes[idx] << " " << confs[idx] <<" " << rects[idx] << endl;
        }
        if (indices.size()) {
            imshow("C",im);
            if (waitKey(10)==27) return 0;
        }
    }
    return 0;
}


int main() {
    return ensemble();
    //return yolo("nano");
    //return yolo("yolo4",true);
    //return test_wbf_2_models();
}