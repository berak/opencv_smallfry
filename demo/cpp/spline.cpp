#include <opencv2/opencv.hpp>
using namespace cv;

using namespace std;
using namespace cv;

int first_click = 0;
int Xholder, Yholder;
Point pt[4];
vector <cv::Point> pts;
Mat img_org;
Mat img_show;// show image
int  Mousedrag = 0;
Mat img;
Mat imgcopy;
int remove1 = 0;
int id = -1;

void onMouse(int event, int x, int y, int flags, void* param)
{
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        for (int i = 0; i < pts.size(); i++)
        {
            if (((x - 4) < pts[i].x && pts[i].x < (x + 4) && (y - 4) < pts[i].y && pts[i].y < (y + 4)))
            {
                id = i;
                break;
            }
            id = -1;
        }

        Mousedrag = 1;
        if (first_click == 4 || first_click == 0)
        {
            if (id == -1)
                pts.push_back(cv::Point(x, y));
            first_click = 1;
        }
        else if (first_click == 1){
            pts.push_back(cv::Point(x, y));
            first_click = 2;
        }
        else if (first_click == 2){
            pts.push_back(cv::Point(x, y));
            first_click = 3;
        }
        else {
            pts.push_back(cv::Point(x, y));
            first_click = 4;
        }
    }
    //in order to delete the last pont
    else if (event == CV_EVENT_RBUTTONDOWN)
    {
        if (first_click == 1)
        {
            pts.pop_back();
            first_click = 4;
        }
        else if (first_click == 2){
            pts.pop_back();
            first_click = 1;
        }
        else if (first_click == 3){
            pts.pop_back();
            first_click = 2;
        }
        else if (first_click == 4){
            pts.pop_back();
            first_click = 3;
        }

    }
    else if (event == CV_EVENT_LBUTTONUP)
    {
        Mousedrag = 0;
        if (id >= 0){
            pts[id].x = x;
            pts[id].y = y;
        }
    }
}


void drawSpline(cv::Point a, cv::Point b, cv::Point c, cv::Point d)
{
    // Way one to assign a matrix
    cv::Mat G_Hermite(4, 2, CV_32FC1);
    G_Hermite.ptr<float>(0)[0] = a.x;
    G_Hermite.ptr<float>(0)[1] = a.y;
    G_Hermite.ptr<float>(1)[0] = b.x;
    G_Hermite.ptr<float>(1)[1] = b.y;
    G_Hermite.ptr<float>(2)[0] = c.x;
    G_Hermite.ptr<float>(2)[1] = c.y;
    G_Hermite.ptr<float>(3)[0] = d.x;
    G_Hermite.ptr<float>(3)[1] = d.y;

    //Way two to assign a matrix
    float data[4][4] = { { -1, 3, -3, 1 }, { 3, -6, 3, 0 }, { -3, 3, 0, 0 }, { 1, 0, 0, 0 } };
    cv::Mat M_Hermite(4, 4, CV_32FC1, &data);
    cv::Mat coefficient_AB(4, 2, CV_32FC1);
    coefficient_AB = M_Hermite * G_Hermite;

    for (float t = 0.0f; t <= 1.0f; t += 0.0001f){
        int x = coefficient_AB.ptr<float>(0)[0] * pow(t, 3) + coefficient_AB.ptr<float>(1)[0] * pow(t, 2)
            + coefficient_AB.ptr<float>(2)[0] * t + coefficient_AB.ptr<float>(3)[0];

        int y = coefficient_AB.ptr<float>(0)[1] * pow(t, 3) + coefficient_AB.ptr<float>(1)[1] * pow(t, 2)
            + coefficient_AB.ptr<float>(2)[1] * t + coefficient_AB.ptr<float>(3)[1];

        cv::circle((img), cv::Point(x, y), 1, cv::Scalar(0, 0, 0), 1);
    }
}

void drawPoint(cv::Point a)
{
    cv::circle(img, a, 1, cv::Scalar(0, 0, 255), 2);
}


int main(int argc, const char * argv[])
{

    imgcopy = Mat(400,400,CV_8UC3, Scalar::all(255));//imread("c:/aa/download.png");
    imgcopy.copyTo(img);

    //Initially set negative number as no points have been selected yet
    pt[0].x = -1;
    pt[0].y = -1;
    pt[1].x = -1;
    pt[1].y = -1;
    pt[2].x = -1;
    pt[2].y = -1;
    pt[3].x = -1;
    pt[3].y = -1;

    cv::namedWindow("test");
    cv::setMouseCallback("test", onMouse);

    while (1)
    {
        imgcopy.copyTo(img);
        for (int i = 4; i <= pts.size(); i = i + 4)
        {
            drawSpline(pts[i - 4], pts[i - 3], pts[i - 2], pts[i - 1]);
        }

        if (!remove1){
            for (int i = 0; i < pts.size(); i++)
                drawPoint(pts[i]);
        }

        cv::imshow("test", img);

        char c = cv::waitKey(1);
        if (c == 27)
            break;
        else if (c == 'R' || c == 'r') //remove all the points

        {
            remove1 = 1;
        }
        else if (c == 'S' || c == 's') // save the iamge
            imwrite("new_test.jpg", img);
    }

    return 0;
}
