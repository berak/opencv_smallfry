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
//
// all-against-all test on att faces
//
int main(int argc, char **argv) {
    PROFILE;
    CommandLineParser parser(argc, argv,
            "{ help h usage ? ||     show this message }"
            "{ siz z          |64|   image size }"
            "{ debug d        ||     show debug images }"
            "{ random r       ||     random convolute }"
            "{ b basedir      |C:/data/fingerprints/DB2_B/|      }"
    );

    if (parser.has("help") || parser.get<String>("basedir").empty()) {
        parser.printMessage();
        return 0;
    }
    int M = 8;
    int N = 10;
    int OFF = 100;
    int Z = parser.get<int>("siz");
    bool randomize = parser.has("random");
    String base = parser.get<String>("basedir");
    DBGDRAW = parser.has("debug");

    cerr << base << " " << Z << " " << randomize << endl;
    Mat_<int> confusion(N,N,0);
    float tp=0,fp=0,tn=0,fn=0;
    const char *fmt("%d_%d.tif");
    for (int p=1; p<=N; p++) { // att starts counting from one ..
        vector<Mat> p1;
        for (int i=1; i<=M; i++) { // images, too. get used to it
            Mat i1 = imread(base + format(fmt, OFF+p, i), 0);
            p1.push_back(i1);
        }
        Ptr<MACE> mace = MACE::create(Z);
        if (randomize) {
            PROFILEX("salt_p")
            mace->salt(p);
        }
        { PROFILEX("train");
        mace->train(p1);
        }
        for (int q=1; q<=N; q++) {
            for (int i=1; i<=M; i++) {
                PROFILEX("process")
                Mat i2 = imread(base + format(fmt, OFF+q, i), 0);
                if (randomize) {
                    PROFILEX("salt_q")
                    mace->salt(q);
                }
                bool same;
                { PROFILEX("same")
                same = mace->same(i2);
                }
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
