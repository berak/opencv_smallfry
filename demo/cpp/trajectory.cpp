#include <opencv2/opencv.hpp>
using namespace cv;

#include <iostream>
#include <fstream>
using namespace std;

#include "profile.h"

const double thresh = 100.0;
const int nbins = 20;
const int tracklen = 15;
const int gridy = 3;
const int gridx = 4;


int extract(const String &filename, Mat &descriptor, int nbins=15, int frameFrom=0, int frameTo=0)
{
    PROFILE;
    double ss = 0.0;

    VideoCapture cap(filename);
    if( !cap.isOpened() )
        return -1;

    cap.set(CAP_PROP_POS_FRAMES, frameFrom);

    vector<vector<Mat>> traj(gridx * gridy);
    Mat frame, gray, prevgray;
    vector<Point2f> pts;
    int nFrames = 0;

    for(;;)
    {
        PROFILEX("loop")
        if ((frameTo != 0) && ((int)cap.get(CAP_PROP_POS_FRAMES) == frameTo))
            break;

        cap >> frame;
        if (frame.empty())
            break;

        {
            PROFILEX("cvt")
            cvtColor(frame, gray, COLOR_BGR2GRAY);
        }
        {
            PROFILEX("res")
            resize(frame,frame,Size(320,240));
        }

        if (! prevgray.empty())
        {
            PROFILEX("proc")
            vector<Mat> features;
            Mat xy[2], diff, mag, ang;
            resize(prevgray - gray, diff, Size(), 0.5, 0.5);
            Sobel(diff, xy[0], CV_32F, 1, 0);
            Sobel(diff, xy[1], CV_32F, 0, 1);
            cartToPolar(xy[0], xy[1], mag, ang);
            for (int R=0; R<gridy; R++)
            {
                for (int C=0; C<gridx; C++)
                {
                    PROFILEX("grid")
                    Mat feat;
                    Size siz(mag.cols/gridx,  mag.rows/gridy);
                    Point cen(C*siz.width + siz.width/2, R*siz.height + siz.height/2);
                    Mat mroi,aroi;
                    getRectSubPix(mag, siz, cen, mroi);
                    getRectSubPix(ang, siz, cen, aroi);
                    Mat mag2 = mroi.mul(mroi);
                    Mat idx;
                    sortIdx(mag2.reshape(1,1),idx,SORT_DESCENDING);
                    for (int i=0; i<nbins; i++)
                    {
                        int r = idx.at<int>(0,i) / mroi.cols;
                        int c = idx.at<int>(0,i) % mroi.cols;
                        feat.push_back(Vec4f(float(c)/mroi.cols, float(r)/mroi.rows, mroi.at<float>(r,c), aroi.at<float>(r,c)));
                    }
                    Mat f = feat.reshape(1,1);
                    features.push_back(f);
                }
            }
            for (size_t i=0; i<features.size(); i++)
            {
                PROFILEX("feature")
                double s = sum(features[i])[0];
                ss += s;
                if (s > thresh)
                {
                    traj[i].push_back(features[i].reshape(1,1));
                    if (traj[i].size() >= tracklen)
                    {
                        Mat des;
                        for (size_t j=0; j<tracklen; j++)
                        {
                            des.push_back(traj[i][j]);
                        }
                        descriptor.push_back(des.reshape(1,1));
                        traj[i].clear();
                    }
                }
                else
                {
                    traj[i].clear();
                }
             }
        }

        nFrames ++;
        cv::swap(prevgray, gray);
    }
    double meanss = ss / (nFrames * gridx * gridy);
    cerr << descriptor.size() << " " << meanss << " " << filename << "          \r";
    return descriptor.total();
 }

int hockey()
{
    Mat data, labels;
    if (1)
    {
        String vids = "c:/data/video/HockeyFights/"; // 500 fi, 500 no
        vector<String> fn;
        glob(vids+"*.avi", fn);

        for (size_t i=0; i<fn.size(); i+=1)
        {
            Mat desc;
            if(extract(fn[i], desc, nbins, 0, 0) < 1)
            {
                continue;
            }
            data.push_back(desc);
            int l = 0;
            if (fn[i].find("fi",0) == vids.size())
            {
                l = 1;
            }
            for (int j=0; j<desc.rows; j++)
            {
                labels.push_back((int)l);
            }
        }
        {
            PROFILEX("save")
            FileStorage fs("hockey_traj.yml.gz", 1);
            fs << "feat" << data;
            fs << "labl" << labels;
        }
    }
    if (data.empty())
    {
        PROFILEX("load")
        FileStorage fs("hockey_traj.yml.gz", 0);
        fs["feat"] >> data;
        fs["labl"] >> labels;
    }
    cerr << endl << "data " << labels.size() << " " << data.size(   ) << endl;
    int NT = 1000; // num tests
    int ROWS = data.rows;
    int COLS = data.cols;

    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setKernel(ml::SVM::LINEAR);
    svm->train(data(Rect(0,NT,COLS, ROWS-2*NT)), 0, labels(Rect(0,NT,1,ROWS-2*NT)));

    Mat rpos,rneg;
    svm->predict(data(Rect(0,0,      COLS,NT)), rpos);
    svm->predict(data(Rect(0,ROWS-NT,COLS,NT)), rneg);
    cerr << countNonZero(rpos==1) << "/" << rpos.rows << endl;
    cerr << countNonZero(rneg==0) << "/" << rneg.rows <<  endl;

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
    if (1)
    {
        String vids = "c:/data/video/ASLAN_AVI/*.avi";
        vector<String> fn;
        glob(vids, fn);

        for (size_t i=0; i<fn.size(); i++)
        {
            Mat desc;
            extract(fn[i], desc, nbins, 0, 0);
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
    return hockey();
}
