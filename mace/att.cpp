#include <math.h>
#include <float.h>
#include <limits.h>
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include "mace.h"

using namespace cv;
using namespace std;


extern bool DBGDRAW;
//
// all-against-all test on att faces
//
int main(int argc, char **argv) {
    CommandLineParser parser(argc, argv,
            "{ help h usage ? ||     show this message }"
            "{ subj s         |40|   num subjects in test(max: 40) }"
            "{ siz z          |64|   image size }"
            "{ debug d        ||     show debug images }"
            "{ random r       ||     random convolute }"
    );
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    int N = parser.get<int>("subj");
    int Z = parser.get<int>("siz");
    bool randomize = parser.has("random");
    DBGDRAW = parser.has("debug");

    String att = "c:/data/faces/att/";
    cerr << att << " " << N  << " " << Z << " " << randomize << endl;
    Mat_<int> confusion(N,N,0);
    float tp=0,fp=0,tn=0,fn=0;
    //Mat predict, truth;
    for (int p=1; p<=N; p++) { // att starts counting from one ..
        vector<Mat> p1;
        for (int i=1; i<=10; i++) {
            Mat i1 = imread(att + format("s%d/%d.pgm", p, i), 0);
            p1.push_back(i1);
        }
        Ptr<MACE> mace = MACE::create(Z, (randomize ? p+123 : 0));
        mace->train(p1);
        //mace->save(format("p%d.xml.gz",p));
        for (int q=1; q<=N; q++) {
            for (int i=1; i<=10; i++) {
                Mat i2 = imread(att + format("s%d/%d.pgm", q, i), 0);
                bool same = mace->same(i2);
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
    if (N<=16)
        cerr << endl << confusion << endl;
    else
        cerr << endl << confusion(Rect(0,0,16,16)) << endl;
    float acc = (tp+tn) / (tp+tn+fp+fn);
    cerr << "tp: " << tp << " fp: " << fp << " tn: " << tn << " fn: " << fn << " acc: " << acc << endl;
    return 1;
}
