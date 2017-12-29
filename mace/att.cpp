#include <math.h>
#include <float.h>
#include <limits.h>
#include <stdio.h>
#include "opencv2/opencv.hpp"
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

    String att = "c:/data/faces/att/*.pgm";
    cerr << att << " " << N  << " " << Z << " " << randomize << endl;
    vector<String> fnames;
    glob(att,fnames,true);

    Mat_<int> confusion(N,N,0);
    float tp=0,fp=0,tn=0,fn=0;
    for (int p=0; p<N; p++) {
        vector<Mat> p1;
        for (int i=0; i<10; i++) {
            String f = fnames[p*10+i];
            Mat i1 = imread(f, 0);
            p1.push_back(i1);
        }
        Ptr<MACE> mace = MACE::create(Z);
        if (randomize)
            mace->salt(fnames[p*10]);
        mace->train(p1);
        for (int q=0; q<N; q++) {
            if (randomize)
                mace->salt(fnames[q*10]);
            for (int i=0; i<10; i++) {
                String f = fnames[q*10+i];
                Mat i2 = imread(f, 0);
                bool same = mace->same(i2);
                if (same) {
                    confusion(p,q) ++; // positively detected.
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
