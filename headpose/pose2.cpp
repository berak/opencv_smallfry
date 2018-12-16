#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"
#include "opencv2/tracking.hpp"


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
    cv::projectPoints(axisPoints,  _rvec, _tvec, _cameraMatrix, _distCoeffs, imagePoints);

    // draw axis lines
    cv::line(_image, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 3);
    cv::line(_image, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 3);
    cv::line(_image, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 3);
}


struct Head {
    cv::Ptr<cv::face::Facemark> facemark;
    std::vector<cv::Point3d> pts3d;
    cv::Mat rvec,tvec;
    cv::Mat camMatrix;

    Head(const cv::String & landmarksModel) {
        facemark = cv::face::createFacemarkLBF();
        facemark->loadModel(landmarksModel);

        if (0) { // actually, we wouldn't need a 3d model, if we'd serialize the corresponding 3d points ..
            cv::FileStorage fs2("points3d.yml",0);
            fs2["points"] >> pts3d;
            fs2.release();
        } else {

            cv::FileStorage fs("data/mdl.yml.gz", cv::FileStorage::READ);
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
            std::vector<cv::Point2d> pts2d;
            cv::Mat meanI = cv::imread("data/reference_320_320.png", 0);
            getkp2d(meanI, pts2d, cv::Rect(80,80, 160,160));

            // get 3d reference points from model
            for(size_t k=0; k<pts2d.size(); k++)
            {
                cv::Vec3d pm = mdl.at<cv::Vec3d>(int(pts2d[k].y), int(pts2d[k].x));
                cv::Point3d p(pm[0], pm[2], -pm[1]);
                pts3d.push_back(p);
            }
            cv::FileStorage fs2("points3d.yml",1);
            fs2 << "points" << pts3d;
            fs2.release();
        }
    }


    void getkp2d(const cv::Mat &I, std::vector<cv::Point2d> &pts2d, const cv::Rect &r) {
        std::vector<cv::Rect> faces(1,r);
        std::vector< std::vector<cv::Point2f> > shapes;
        if (! facemark->fit(I,faces,shapes))
            return;

        pts2d.clear();
        for(size_t k=0; k<shapes[0].size(); k++)
        {
            pts2d.push_back(shapes[0][k]);
        }
    }

    cv::Mat pnp(const cv::Size &s, std::vector<cv::Point2d> &pts2d) {
        // camMatrix based on img size
        int max_d = std::max(s.width,s.height);
        camMatrix = (cv::Mat_<double>(3,3) <<
            max_d,   0, s.width/2.0,
            0,     max_d, s.height/2.0,
            0,   0,        1.0);

        // 2d -> 3d correspondence
        cv::solvePnP(pts3d, pts2d, camMatrix, cv::Mat(1,4,CV_64F,0.0), rvec, tvec, false, cv::SOLVEPNP_EPNP);

        // get 3d rot mat
        cv::Mat rotM(3, 3, CV_64F);
        cv::Rodrigues(rvec, rotM);

        // push tvec to transposed Mat
        cv::Mat rotMT = rotM.t();
        rotMT.push_back(tvec.reshape(1, 1));

        // transpose back, and multiply
        return camMatrix * rotMT.t();
    }

    void draw(cv::Mat &I, const std::vector<cv::Point2d> &pts, const cv::Rect &r)
    {
        for (size_t i=0; i<pts.size()-1; i++)
        {
            //cv::line(I,pts[i],pts[i+1],cv::Scalar(200));
            cv::circle(I,pts[i],3,cv::Scalar(0,200));
        }
        drawAxis(I, camMatrix, cv::Mat(1,4,CV_64F,0.0), rvec, tvec, 80);
    }
};

int main() {
    cv::String landmarksModel = "c:/data/mdl/lbfmodel.yaml";
    cv::String cascade_name   = "c:/data/mdl/haarcascade_frontalface_alt.xml";
    cv::VideoCapture cap(0);
    cv::Ptr<cv::Tracker> mosse;
    cv::CascadeClassifier face_cascade;
    face_cascade.load(cascade_name);
    Head head(landmarksModel);
    int fno=0;
    while(cap.isOpened()) {
        cv::Mat frame;
        if (! cap.read(frame))
            break;
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        equalizeHist(gray, gray);
        std::vector<cv::Point2d> pts2d; // 2d model from landmarks
        cv::Rect box;
        if (fno %100 == 0) { // cascade detection is very noisy, tracking makes it more stable
            std::vector<cv::Rect> faces;
            face_cascade.detectMultiScale(gray, faces, 1.4, 2, cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
            if (faces.size()) {
                box = faces[0];
                cv::Rect2d _box(box);
                mosse = cv::TrackerMOSSE::create();
                mosse->init(gray, _box);
                std::cout << ".";
            }
        } else {
            if (!mosse.empty()) {
                cv::Rect2d _box;
                mosse->update(gray, _box);
                box = _box;
            }
        }
        if (! box.empty()) {
            head.getkp2d(gray, pts2d, box);
            if (pts2d.size()) {
                cv::Mat p = head.pnp(frame.size(), pts2d);
                head.draw(frame, pts2d, box);
            }
        }
        cv::imshow("box",frame);
        int k = cv::waitKey(10);
        if (k == 27)   break;
        fno++;
    }
    return 0;
}
