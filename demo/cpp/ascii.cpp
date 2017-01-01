#include <stdio.h>
#include <windows.h>
#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;

void ascii(const Mat &m) {
    int S = 10;
    String ramp = " .:-=+*#%@";
    //String ramp = "$@B§8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";
    //const static char dith[S] = {' ','_','.',',',':',';','-','+','*','#'};

    String linie;
    for (int r=0; r<m.rows; r++) {
        for ( int c=0; c<m.cols; c++) {
            uchar v = m.at<uchar>(r,c)/(256/S);
            linie += ramp[ v % S ];
        }
        linie += "\r\n";
    }
    cerr << linie << endl;
}

int main_loop(String vid)
{
    int     H=30,
            W=80;

    namedWindow("gr",0);
    VideoCapture cap;
    if (vid=="") cap.open(0);
    else cap.open(vid);
    while(cap.isOpened())
    {
        Mat f;
        cap >> f;
        if (f.empty())
        {
            if (vid!="")
            {
                cap.set(CAP_PROP_POS_FRAMES, 0);
                continue;
            }
            else
            {
                break;
            }
        }
        Mat r, g;
        cvtColor(f,g,COLOR_BGR2GRAY);
        resize(g,r,Size(W,H));
        imshow("ocv",f);
        imshow("gr",r);
        int k = waitKey(20);
        if (k==27) break;
        ascii(r);
    }
    return 0;
}

int main(int argc, char **argv) {
    //HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    //CONSOLE_SCREEN_BUFFER_INFO consoleInfo;
    //WORD saved_attributes;

    /* Save current attributes */
    //GetConsoleScreenBufferInfo(hConsole, &consoleInfo);
    //saved_attributes = consoleInfo.wAttributes;

    //SetConsoleTextAttribute(hConsole, FOREGROUND_BLUE | FOREGROUND_INTENSITY);
    //printf("This is some nice COLORFUL text, isn't it?");

    main_loop(argc>1?argv[1]:"");

    /* Restore original attributes */
    //SetConsoleTextAttribute(hConsole, saved_attributes);
    //printf("Back to normal");

    return 0;
}
