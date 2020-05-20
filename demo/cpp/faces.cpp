#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    theRNG().state = getTickCount();

    float confThreshold = 0.9;
    String c_in("c:\\data\\cache\\15\\*.png");
    String c_out("c:\\data\\faces\\unsorted\\");
    dnn::Net net = dnn::readNet("c:/data/dnn/res10_300x300_ssd_iter_140000.caffemodel", "frec.prototxt");

    vector<String> fn;
    glob(c_in, fn, true);
    cout << fn.size() << " files." << endl;
    for(size_t i=0; i<fn.size(); i++)
    {
        Mat im = imread(fn[i]);
        CV_Assert(! im.empty());
        Rect bounds(0,0,im.cols,im.rows);

        Mat blob = dnn::blobFromImage(im, 1, Size(300,300), Scalar(104, 177, 123, 0), false, false);
        net.setInput(blob);
        Mat res = net.forward("detection_out");
        Mat faces(res.size[2],res.size[3], CV_32F, res.ptr<float>());
        for (int j=0; j<faces.rows; j++)
        {
            float *data = faces.ptr<float>(j);
            float confidence = data[2];
            if (confidence > confThreshold)
            {
                int left   = (int)(data[3] * im.cols);
                int top    = (int)(data[4] * im.rows);
                int right  = (int)(data[5] * im.cols);
                int bottom = (int)(data[6] * im.rows);
                Rect r(Point(left,top), Point(right,bottom));
                r &= bounds;
                cout << fn[i] << " " << confidence << " " << r << endl;
                int64 k = theRNG().uniform(0,100000000);
                imwrite(c_out + format("%08x.png",k),im(r));
            }
        }
    }
    return 0;
}
