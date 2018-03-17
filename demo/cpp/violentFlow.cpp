#include <opencv2/opencv.hpp>
using namespace cv;

#include <iostream>
#include <fstream>
using namespace std;

//
// http://www.openu.ac.il/home/hassner/data/violentflows/violent_flows.pdf
//

Mat normalizedHist(const Mat &roi, int nbins)
{
    Mat_<float> hist(1, nbins, 0.0f);
    Mat n;
    normalize(roi, n, nbins-1, 0, NORM_MINMAX);
    for (size_t j=0; j<n.rows; j++)
    {
        float *p = n.ptr<float>(j);
        for (size_t i=0; i<n.cols; i++)
        {
            hist(int(p[i])) ++;
        }
    }
    normalize(hist, hist);
    return hist;
}

void accum(Mat &cur, Mat &prev, Mat &acc)
{
    if (! prev.empty())
    {
        Mat diff = cur - prev;

        double M;
        minMaxLoc(diff, 0, &M, 0, 0);

        Mat bin;
        threshold(diff, bin, M/8, 1.0, 0);

        if (acc.empty())
            acc = Mat(cur.size(), CV_32F, 0.0f);

        acc += bin;
    }
    std::swap(prev, cur);
}

int violentFlow(const String &filename, Mat &descriptor, int nbins=20, int frameFrom=0, int frameTo=0)
{
    VideoCapture cap(filename);
    if( !cap.isOpened() )
        return -1;

    cap.set(CAP_PROP_POS_FRAMES, frameFrom);

    Mat flow, frame;
    Mat gray, prevgray;
    Mat mag, prevmag, accmag;
    Mat ang, prevang, accang;
    int nFramesInMag = 0;

    for(;;)
    {
        if ((frameTo != 0) && ((int)cap.get(CAP_PROP_POS_FRAMES) == frameTo))
            break;

        cap >> frame;
        if (frame.empty())
            break;

        resize(frame,frame,Size(320,240));
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        if (! prevgray.empty())
        {
            Mat xy[2];
            if (1) {
                Mat diff = prevgray - gray;
                Sobel(diff, xy[0], CV_32F, 1, 0);
                Sobel(diff, xy[1], CV_32F, 0, 1);
            } else {
                calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 2, 9, 3, 5, 1.2, 0);
                split(flow, xy);
            }
            cartToPolar(xy[0], xy[1], mag, ang);
        }
        accum(mag, prevmag, accmag);
        accum(ang, prevang, accang);

        nFramesInMag ++;
        char c = ' ' + (nFramesInMag % 3);
        cerr << c << '\r';
        std::swap(prevgray, gray);
    }

    Mat hist;
    int h = accmag.rows/4;
    int w = accmag.cols/4;
    for (int r=0; r<accmag.rows; r+=h)
    {
        for (int c=0; c<accmag.cols; c+=w)
        {
            Rect roi(c,r,w,h);
            hist.push_back(normalizedHist(accmag(roi),nbins));
            hist.push_back(normalizedHist(accang(roi),nbins));
        }
    }
    descriptor = hist.reshape(1,1);
    return 0;
}

int hockey()
{
    Mat data;
    if (0)
    {
        String vids = "c:/data/video/HockeyFights/*.avi"; // 500 fi, 500 no
        vector<String> fn;
        glob(vids, fn);

        int nbins = 32;
        for (size_t i=0; i<fn.size(); i++)
        {
            Mat desc;
            if (-1 == violentFlow(fn[i], desc, nbins, 0, 0))
                continue;
            data.push_back(desc);
            cerr << fn[i] << endl;
        }
        FileStorage fs("hockey_fbflow.yml.gz", 1);
        fs << "sobel" << data;
    }

    FileStorage fs("hockey_fbflow.yml.gz", 0);
    fs["sobel"] >> data;
    int NT = 100; // num tests
    int COLS = data.cols;
    Mat labels(1000-2*NT,1,CV_32S, Scalar(1)); // keep first & last 50 for testing
    labels(Rect(0,500-NT,1,500-NT)) = 0;

    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setKernel(ml::SVM::LINEAR);
    svm->train(data(Rect(0,NT,COLS,1000-2*NT)), 0, labels);

    Mat rpos,rneg;
    svm->predict(data(Rect(0,0,      COLS,NT)), rpos);
    svm->predict(data(Rect(0,1000-NT,COLS,NT)), rneg);
    cerr << countNonZero(rpos==1) << endl;
    cerr << countNonZero(rneg==0) << endl;

    return 0;
}

Mat dist(const Mat &a, const Mat &b)
{
    Mat c = a - b;
    multiply(c,c,c);
    return c; // (a-b)^2
}

int aslan()
{
    Mat data;
    if (0)
    {
        String vids = "c:/data/video/ASLAN_AVI/*.avi";
        vector<String> fn;
        glob(vids, fn);

        int nbins = 32;
        for (size_t i=0; i<fn.size(); i++)
        {
            Mat desc;
            violentFlow(fn[i], desc, nbins, 0, 0);
            data.push_back(desc);
            cerr << fn[i] << endl;
        }
        FileStorage fs("aslan_sobel.yml.gz", 1);
        fs << "feat" << data;
    }

    FileStorage fs("aslan_sobel.yml.gz", 0);
    fs["feat"] >> data;
    fs.release();
    cerr << data.size() << " features." << endl;
    float acc = 0.0f;
    for (int split=0; split<10; split++)
    {
        Mat train_s, train_l;
        Mat test_s,  test_l;
        ifstream tp("view2.txt");
        int k=0, i1, i2, same, dummy;
        while(tp >> i1 >> i2 >> same >> dummy >> dummy) {
            if ((k >= split*600) && (k < (split+1)*600)) {
                test_s.push_back(dist(data.row(i1-1), data.row(i2-1)));
                test_l.push_back(same);
            } else {
                train_s.push_back(dist(data.row(i1-1), data.row(i2-1)));
                train_l.push_back(same);
            }
            k++;
        }

        Ptr<ml::SVM> svm = ml::SVM::create();
        svm->setKernel(ml::SVM::LINEAR);
        svm->train(train_s, 0, train_l);

        Mat r;
        svm->predict(test_s, r);
        r.convertTo(r, CV_32S);
        float a = (float)countNonZero(r==test_l)/test_s.rows;
        cerr << "split " << split << "  " << a << endl;
        acc += a;
    }
    cerr << "final    " << acc*0.1f << endl;
    return 0;
}

int main(int argc, char** argv)
{
    return aslan();
}
