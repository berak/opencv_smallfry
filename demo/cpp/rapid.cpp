#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/rapid.hpp"
using namespace cv;
#include <iostream>
using namespace std;

int main()
{
    // a unit sized box
    std::vector<Vec3f> vtx = {
        {1, -1, -1}, {1, -1, 1}, {-1, -1, 1}, {-1, -1, -1}, {1, 1, -1}, {1, 1, 1}, {-1, 1, 1}, {-1, 1, -1},
    };
    std::vector<Vec3i> tris = {
        {2, 4, 1}, {8, 6, 5}, {5, 2, 1}, {6, 3, 2}, {3, 8, 4}, {1, 8, 5},
        {2, 3, 4}, {8, 7, 6}, {5, 6, 2}, {6, 7, 3}, {3, 7, 8}, {1, 4, 8},
    };
    Mat(tris) -= Scalar(1, 1, 1);

    Vec3f rot(0,0,0), trans(0,0,0);
    Mat frame = imread("box.jpg");
    Mat_<float> K(3,3); K <<
        .15 * frame.cols, 0, frame.cols/2,
        0, .15 * frame.rows, frame.rows/2,
        0, 0, 1;
    cout << frame.size() << K << endl;
    Mat gray; cvtColor(frame,gray,COLOR_BGR2GRAY);
    for(int i = 0; i < 3; i++) {// do two iteration
        rapid::rapid(gray, 100, 20, vtx, tris, K, rot, trans);
    }
    cout << trans << " " << rot << endl;

    Mat pts2d;
    projectPoints(vtx, rot, trans, K, noArray(), pts2d);
    rapid::drawWireframe(frame, pts2d, tris, Scalar(200), LINE_8);
    imshow("R",frame);
    waitKey();
    return 0;
}
/*
int main()
{
    // a unit sized box
    std::vector<Vec3f> vtx = {
        {1, -1, -1}, {1, -1, 1}, {-1, -1, 1}, {-1, -1, -1}, {1, 1, -1}, {1, 1, 1}, {-1, 1, 1}, {-1, 1, -1},
    };
    std::vector<Vec3i> tris = {
        {2, 4, 1}, {8, 6, 5}, {5, 2, 1}, {6, 3, 2}, {3, 8, 4}, {1, 8, 5},
        {2, 3, 4}, {8, 7, 6}, {5, 6, 2}, {6, 7, 3}, {3, 7, 8}, {1, 4, 8},
    };
    Mat(tris) -= Scalar(1, 1, 1);

    Vec3f rot(0,0,0), trans(0,0,0);
    VideoCapture cap(0);
    int j=0, k=0;
    while(1) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        Mat_<float> K(3,3); K <<
            frame.cols, 0, frame.cols/2,
            0, frame.rows, frame.rows/2,
            0, 0, 1;
        cout << frame.size() << K << endl;
        Mat gray; cvtColor(frame,gray,COLOR_BGR2GRAY);
        if (k==' ') {
            for(int i = 0; i < 3; i++) {// do two iteration
                rapid::rapid(gray, 100, 20, vtx, tris, K, rot, trans);
            }
            cout << (j++) << " " << trans << " " << rot << endl;
            Mat pts2d;
            projectPoints(vtx, rot, trans, K, noArray(), pts2d);
            rapid::drawWireframe(frame, pts2d, tris, Scalar(200), LINE_8);
        }
        imshow("R",frame);
        k = waitKey(10);
        if (k == 27) break;
    }
    return 0;
}
*/
/*int main()
{
    // a unit sized box
    std::vector<Vec3f> vtx = {
        {1, -1, -1}, {1, -1, 1}, {-1, -1, 1}, {-1, -1, -1}, {1, 1, -1}, {1, 1, 1}, {-1, 1, 1}, {-1, 1, -1},
    };
    std::vector<Vec3i> tris = {
        {2, 4, 1}, {8, 6, 5}, {5, 2, 1}, {6, 3, 2}, {3, 8, 4}, {1, 8, 5},
        {2, 3, 4}, {8, 7, 6}, {5, 6, 2}, {6, 7, 3}, {3, 7, 8}, {1, 4, 8},
    };
    Mat(tris) -= Scalar(1, 1, 1);

    // camera setup
    Size sz(1280, 720);

    Mat K = getDefaultNewCameraMatrix(Matx33f::diag(Vec3f(800, 800, 1)), sz, true);
    cout << "K " << K << endl;
    Vec3f trans = {0, 0, 5};
    Vec3f rot = {0.7f, 0.6f, 0};
    Vec3f rot0(rot);
    cout << "t " << trans << " " << rot0 << endl;

    // draw something
    Mat_<uchar> img(sz, uchar(0));

    Mat pts2d;
    projectPoints(vtx, rot, trans, K, noArray(), pts2d);
    rapid::drawWireframe(img, pts2d, tris, Scalar(120), LINE_8);

    // recover pose form different position
    Vec3f t_init = Vec3f(0.1f, 0, 5);

    pts2d.release();
    projectPoints(vtx, rot, t_init, K, noArray(), pts2d);
    rapid::drawWireframe(img, pts2d, tris, Scalar(180), LINE_8);

    cout << "i " << t_init << " " << rot << endl;
    for(int i = 0; i < 3; i++) {// do two iteration
        rapid::rapid(img, 100, 20, vtx, tris, K, rot, t_init);
        cout << i << " " << t_init << " " << rot << endl;
    }

    pts2d.release();
    projectPoints(vtx, rot, t_init, K, noArray(), pts2d);
    rapid::drawWireframe(img, pts2d, tris, Scalar(255), LINE_8);

    // assert that it improved from init
    double nt = cv::norm(trans - t_init);
    double nr = cv::norm(rot - rot0);
    cout << "T " << nt << " R " << nr << endl;
    imshow("R",img);
    waitKey();
    return 0;
}
*/