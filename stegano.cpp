#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "time.h"

using namespace cv;
using namespace std;

void stega_write(Mat chan, char*txt, int off=0) {
    for ( int i=0; i<strlen(txt); i++ ) {
        for ( int j=0; j<8; j++ ) {
            uchar &c = chan.at<uchar>(off);
            c &= ~1;
            c |= (txt[i]&(1<<j)) ? 1 : 0;
            off ++;
        }
    }
}

void stega_read(Mat chan, char*txt, int off=0) {
    while ( 1 ) {
        uchar t = 0;
        for ( int j=0; j<8; j++ ) {
            uchar c = chan.at<uchar>(off);
            t |= ((c&1)&0x01) << j;
            off ++;
        }
        *txt ++ = t;
        if ( t == 0 )
            break;
    }
}

int main( int argc, char **argv ) {

    // crypt it:
    char *txt =  "JimmyCarter: in case you're still interested";
    Mat input = imread("Clipboard03.png");
    Mat chan[3]; split(input,chan);

    stega_write(chan[0],txt,0);

    Mat output;  merge(chan,3,output);
    imwrite("stega.png",output);
    cout << " orig: " << txt << endl;


    // decrypt it:
    input = imread("stega.png");
    split(input,chan);
    char txt2[400] = {0};

    stega_read(chan[0],txt2,0);

    cout << " steg: " << txt2 << endl;
    imshow("out",input);
    waitKey(0);
}


