#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

bool down   = false;
bool hasRoi = false;
Point tl,br;
Rect  roi;

void onmouse( int evnt, int x, int y, int d, void *p )
{
    if ( evnt==4 )   
    {
        if ( tl.x<br.x+5)
            roi=Rect(tl,br);   
        down = false; 
    }
    if ( evnt==1 )   { tl = Point(x,y); down = true; }
    if ( d==1 )      { br = Point(x,y); }
}
int main(int argc, char *argv[]) 
{
    namedWindow("win",1);
    setMouseCallback("win",onmouse);
    VideoCapture cap(0); 
    while(cap.isOpened())
    {
        Mat imag, show;
        cap >> imag;

        if ( imag.empty() )
            break;
        if ( roi.area()>=32 ) 
            resize(imag(roi),show,imag.size());
        else 
            show = imag;

        if ( down )
            rectangle(show,tl,br,Scalar(0,0,200));

        int k = waitKey(40);
        if ( k==27  ) break;
        if ( k==' ' ) roi.x=roi.y=roi.height=roi.width=0; // revert to original

        imshow("win",show);
    }
    return 0;
}
