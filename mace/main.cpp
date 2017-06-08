#include <math.h>
#include <float.h>
#include <limits.h>
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "mace.h"

using namespace cv;
using namespace std;

//
// all-against-all test on att faces
//
int main() {
    const float thresh = 0.3f; // derived by experiment
    const int N = 40; // 40 subjects.

    Ptr<MACE> mace = createMACE(80);
    String att = "c:/data/faces/att/";
    Mat_<int> confusion(N,N,0);
    for (int p=1; p<=N; p++) { // att starts counting from one ..
        vector<Mat> p1;
        for (int i=1; i<=10; i++) {
            Mat i1 = imread(att + format("s%d/%d.pgm", p, i), 0);
            p1.push_back(i1);
        }
        mace->compute(p1);
        for (int q=1; q<=N; q++) {
            for (int i=1; i<=10; i++) {
                Mat i2 = imread(att + format("s%d/%d.pgm", q, i), 0);
                double pred = mace->correlate(i2);
                if (pred > thresh) {
                    confusion(p-1,q-1) ++; // positively detected.
                }
            }
        }
        cerr << ".";
    }
    if (N<=16)
        cerr << endl << confusion << endl;
    else
        cerr << endl << confusion(Rect(0,0,16,16)) << endl;

    float pos = sum(confusion)[0];
    float tp  = sum(confusion.diag())[0];
    float fp  = pos - tp;
    float fn  = confusion.cols*10 - tp;
    float tn  = confusion.total() - pos;
    float acc = (tp+tn) / confusion.total();
    cerr << pos << " " << tp << " " << fp << " " << fn << " " << acc << endl;
    return 1;
}
