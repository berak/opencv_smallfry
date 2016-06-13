#include "xui.h"
#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    Ptr<UI> ui = createUI("vid-ui");
    int h=30, k2=13, k3=99, k4=1, k5=0, k6=0, k7=2;
    ui->addTrigger("info",Rect(30,h,150,30),   Scalar(20,80,20), k6);
    ui->addSlider("heat", Rect(30,h*2,150,30), Scalar(0,0,170),  k2);
    ui->addSlider("water",Rect(30,h*3,150,30), Scalar(50,0,120), k3);
    ui->addButton("candy",Rect(30,h*4,150,30), Scalar(120,0,20), k4);
    ui->addButton("lm22", Rect(30,h*5,150,30), Scalar(120,0,20), k5);
    vector<String> ch = {"alpha","beta","cream","dino","eps","full","gaga"};
    ui->addChoice(ch,     Rect(30,h*6,150,30), Scalar(20,40,20), k7);

    VideoCapture cap(0);
    while(cap.isOpened())
    {
        Mat frame;
        cap >> frame;
        ui->show(frame);
        int k = waitKey(40);
        if (k==27) break;
        if (k=='1') ui->setText(0,"1111");
        if (k=='2') ui->setText(0,"2222");
        if (k=='\t') ui->toggle();
        if (k6) { cerr << ui->info() << endl; k6=0; }
    } 
    
    return 0;
}
