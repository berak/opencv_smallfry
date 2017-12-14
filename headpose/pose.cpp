#include "opencv2/opencv.hpp"
#include <cstdio>
#include <vector>
#include <deque>
#include <iostream>


#include "opencv2/core/core_c.h" // shame, but needed for using dlib
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>

#include "pico/picornt.hpp"

using namespace std;

//
// stolen from charuco ..
//
void drawAxis(cv::InputOutputArray _image, cv::InputArray _cameraMatrix, cv::InputArray _distCoeffs,
              cv::InputArray _rvec, cv::InputArray _tvec, float length) {

    CV_Assert(_image.getMat().total() != 0 &&
              (_image.getMat().channels() == 1 || _image.getMat().channels() == 3));
    CV_Assert(length > 0);

    // project axis points
    std::vector< cv::Point3f > axisPoints;
    axisPoints.push_back(cv::Point3f(0, 0, 0));
    axisPoints.push_back(cv::Point3f(length, 0, 0));
    axisPoints.push_back(cv::Point3f(0, length, 0));
    axisPoints.push_back(cv::Point3f(0, 0, length));
    std::vector< cv::Point2f > imagePoints;
    cv::projectPoints(axisPoints, _rvec, _tvec, _cameraMatrix, _distCoeffs, imagePoints);

    // draw axis lines
    cv::line(_image, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 3);
    cv::line(_image, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 3);
    cv::line(_image, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 3);
}


//
// e.g. to smooth landmarks
//
template<class T>
struct Ipol
{
    size_t N;
    std::deque<T> q;

    Ipol(size_t n=10) : N(n) {}

    T operator ()(T t)
    {
        q.push_back(t);
        if (q.size() > N)
            q.pop_front();

        T acc(0,0);// = 0;
        for (size_t i=0; i<q.size(); ++i)
            acc += q[i];
        return acc / double(q.size());
    }
};

struct Head
{
    dlib::shape_predictor sp;
    vector<cv::Point3d> pts3d;
    vector<Ipol <cv::Point2d> > ipol;
    cv::Mat rvec,tvec;
    cv::Mat camMatrix;

    Head()
    {
        dlib::deserialize("c:/p/ocv/old/data/shape_predictor_68_face_landmarks.dat") >> sp;

        cv::FileStorage fs("c:/p/ocv/old/data/mdl.yml.gz", cv::FileStorage::READ);
        cv::Mat mdl;
        fs["mdl"] >> mdl;
        fs.release();

        //// if you want to see the 3d model ..
        //cv::Mat ch[3];
        //cv::split(mdl, ch);
        //cv::Mat_<double> depth;
        //cv::normalize(ch[1], depth, -100);
        //cv::imshow("head1", depth);

        // get 2d reference points from image
        vector<cv::Point2d> pts2d;
        cv::Mat meanI = cv::imread("c:/p/ocv/old/data/reference_320_320.png", 0);
        getkp2d(meanI, pts2d, cv::Rect(80,80, 160,160), false);

        // get 3d reference points from model
        for(size_t k=0; k<pts2d.size(); k++)
        {
            cv::Vec3d pm = mdl.at<cv::Vec3d>(int(pts2d[k].y), int(pts2d[k].x));
            cv::Point3d p(pm[0], pm[2], -pm[1]);
            pts3d.push_back(p);
        }
    }


    void getkp2d(const cv::Mat &I, vector<cv::Point2d> &pts2d, const cv::Rect &r, int interpolate=0)
    {
        dlib::rectangle rec(r.x, r.y, r.x+r.width, r.y+r.height);
        dlib::full_object_detection shape = sp(dlib::cv_image<uchar>(I), rec);

        pts2d.clear();
        for(size_t k=0; k<shape.num_parts(); k++)
        {
            cv::Point2d p(shape.part(k).x(), shape.part(k).y());
            pts2d.push_back(p);
        }
        if (interpolate > 0)
        {
            if (ipol.empty() || (ipol[0].N != interpolate))
                ipol = vector<Ipol <cv::Point2d> >(pts2d.size(), interpolate);

            for(size_t k=0; k<pts2d.size(); k++)
            {
                pts2d[k] = ipol[k](pts2d[k]);
            }
        }
    }

    cv::Mat pnp(const cv::Size &s, vector<cv::Point2d> &pts2d)
    {
        // camMatrix based on img size
        int max_d = std::max(s.width,s.height);
        camMatrix = (cv::Mat_<double>(3,3) <<
            max_d,   0, s.width/2.0,
            0,     max_d, s.height/2.0,
            0,   0,        1.0);

        // 2d -> 3d correspondence
        cv::solvePnP(pts3d, pts2d, camMatrix, cv::Mat(1,4,CV_64F,0.0), rvec, tvec, false, cv::SOLVEPNP_EPNP);
        //cerr << "rot " << rvec.t() *180/CV_PI << endl;
        //cerr << "tra " << tvec.t() << endl;

        // get 3d rot mat
        cv::Mat rotM(3, 3, CV_64F);
        cv::Rodrigues(rvec, rotM);

        // push tvec to transposed Mat
        cv::Mat rotMT = rotM.t();
        rotMT.push_back(tvec.reshape(1, 1));

        // transpose back, and multiply
        return camMatrix * rotMT.t();
    }
    void draw(cv::Mat &I, const vector<cv::Point2d> &pts, const cv::Rect &r)
    {
        for (size_t i=0; i<pts.size()-1; i++)
        {
            cv::line(I,pts[i],pts[i+1],cv::Scalar(200));
            cv::circle(I,pts[i],3,cv::Scalar(0,200));
        }
        drawAxis(I, camMatrix, cv::Mat(1,4,CV_64F,0.0), rvec, tvec, 80);
    }
};

int main()
{
    cerr << "press 'esc.' to stop, "<<endl<<"\t'd' to decrease or, 'D' to increase landmark interpolation." << endl;

    Head head;
    cv::VideoCapture cap(0);
    int denoise = 5;
    while(cap.isOpened())
    {
        cv::Mat frame;
        if (! cap.read(frame))
            break;
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        vector<cv::Point2d> pts2d; // 2d model from landmarks
        vector<cv::Vec4f>   obj;   // pico object: (confidence, x, y, width)
        int n = pico::findObjects(gray, obj);
        if (n > 0)
        {
            const cv::Vec4f &o = obj[0];
            float conf = o[0];
            if (conf < 4.0f) continue;
            int w = int(o[3])/2; // half width
            cv::Rect r(int(o[2])-w, int(o[1])-w, 2*w,2*w);
            //cv::rectangle(frame, r, cv::Scalar(0, 4*conf, 0), 1);

            head.getkp2d(gray, pts2d, r, denoise);
            cv::Mat p = head.pnp(frame.size(), pts2d);
            head.draw(frame, pts2d, r);
        }

        cv::imshow("box",frame);
        int k = cv::waitKey(10);
        if (k == 27)   break;
        if (k == 'd') if(denoise>0) denoise--, cerr << "denoise " << denoise << endl;
        if (k == 'D') denoise++, cerr << "denoise " << denoise << endl;
    }
    return 0;
}
