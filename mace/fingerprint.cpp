#include <math.h>
#include <float.h>
#include <limits.h>
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include "mace.h"
#include "profile.h"

using namespace cv;
using namespace std;


extern bool DBGDRAW;
bool nodeskew = true;

Mat deskew(const Mat &img)
{
    if (nodeskew==true) return img;
    PROFILEX("deskew");
    int SZ = img.rows;
    Moments m = moments(~img);
    if (abs(m.mu02) < 1e-2) {
        cerr << "too small mu02 " << m.mu02 << endl;
        return img;
    }
    double skew = m.mu11 / m.mu02;
    double skale = 1.0 - double(img.cols)/m.m00;
    Point2f obj(m.m10/m.m00, m.m01/m.m00);
    double angle = (360 * (skew)) / CV_PI; // atan2(p.y,p.x);
    Mat m2 = getRotationMatrix2D(obj, angle, skale);
    if (DBGDRAW) cerr << angle << " " << skale << " " << m2 << endl;
    Mat res;
    warpAffine(img, res, m2, Size(SZ,SZ), WARP_INVERSE_MAP | INTER_LINEAR);
    //warpAffine(img, res, m2, Size(SZ,SZ));
    return res;
}

int main(int argc, char **argv) {
    PROFILE;
    CommandLineParser parser(argc, argv,
            "{ help h usage ? ||     show this message }"
            "{ siz z          |70|   image size }"
            "{ debug d        ||     show debug images }"
            "{ random r       ||     random convolute }"
//            "{ m multi        ||     use 4 maze detectors }"
            "{ n nodeskew     ||     disable deskewing of input image }"
            "{ b basedir      |C:/data/fingerprints/DB4_B/|      }"
    );

    if (parser.has("help") || parser.get<String>("basedir").empty()) {
        parser.printMessage();
        return 0;
    }
    int M = 8; // items per class
    int N = 10; // classes
    int OFF = 100; // "109_3.tif", both numbers 1 based
    int Z = parser.get<int>("siz");
    bool randomize = parser.has("random");
  //  bool multi = parser.has("multi");
    String base = parser.get<String>("basedir");
    DBGDRAW = parser.has("debug");
    nodeskew = parser.has("nodeskew");

    cerr << base << " " << Z << " " << randomize << " " << nodeskew << endl;
    Mat_<int> confusion(N,N,0);
    float tp=0,fp=0,tn=0,fn=0;
    const char *fmt("%d_%d.tif");
    for (int p=1; p<=N; p++) { // this db starts counting from one ..
        vector<Mat> p1;
        for (int i=1; i<=M; i++) {
            Mat i1 = deskew(imread(base + format(fmt, OFF+p, i), 0));
            p1.push_back(i1);
        }
        Ptr<MACE> mace;
        /*if (multi) {
            mace = MACE::createSampler(Z,vector<Rect2d>{
                Rect2d(0,0,1,1),           // whole image
                Rect2d(0.25,0.25,0.5,0.5), // center
                Rect2d(0.1,0.1,0.5,0.5),   // top left
                Rect2d(0.2,0.3,0.5,0.5)    // bot something
            });
        } else {*/
        mace = MACE::create(Z);
        //}
        if (randomize) {
            mace->salt(p);
        }
        mace->train(p1);
        for (int q=1; q<=N; q++) {
            for (int i=1; i<=M; i++) {
                PROFILEX("process")
                Mat i2;
                { PROFILEX("read")
                i2 = deskew(imread(base + format(fmt, OFF+q, i), 0));
                }
                if (randomize) {
                    mace->salt(q);
                }
                bool same;
                same = mace->same(i2);
                if (same) {
                    confusion(p-1,q-1) ++; // positively detected.
                    if (p==q)  tp ++;
                    else       fp ++;
                } else {
                    if (p==q)  fn ++;
                    else       tn ++;
                }
            }
        }
        cerr << ".";
    }
    if (N <= 16)
        cerr << endl << confusion << endl;
    else
        cerr << endl << confusion(Rect(0,0,16,16)) << endl;
    float all = (tp+tn+fp+fn);
    float acc = (tp+tn) / all;
    cerr << "tp: " << tp << " fp: " << fp << " tn: " << tn << " fn: " << fn << " all: " << all << " acc: " << acc << endl;
    return 0;
}
