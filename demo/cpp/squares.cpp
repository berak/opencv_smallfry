
// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <math.h>
#include <string.h>

using namespace cv;
using namespace std;

static void help()
{
    cout <<
    "\nA program using pyramid scaling, Canny, contours, contour simpification and\n"
    "memory storage (it's got it all folks) to find\n"
    "squares in a list of images pic1-6.png\n"
    "Returns sequence of squares detected on the image.\n"
    "the sequence is stored in the specified memory storage\n"
    "Call:\n"
    "./squares [file_name (optional)]\n"
    "Using OpenCV version " << CV_VERSION << "\n" << endl;
}


int thresh = 50, N = 11;
const char* wndname = "Square Detection Demo";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares( const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for( int c = 1; c < 2; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
                {
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.5 )
                        squares.push_back(approx);
                }
            }
        }
    }
}

//
//  0 1
//  3 2
//
int index_cw(Point q, const vector<Point> &pt) {
	int left=0, top=0;
	for (Point p:pt) {
		if (q.x < p.x) left++;
		if (q.y < p.y) top++;
	}
	return (left>=2 && top>=2) ? 0:
	       (left<2 && top>=2) ? 1:
	       (left<2 && top<2) ? 2 : 3;
}
vector<Point2f> sort_cw(const vector<Point> &pt) {
	vector<Point2f> npt(pt.size());
	for (Point p:pt) {
		int i = index_cw(p,pt);
		npt[i] = p;
	}
	return npt;
}


int cluster(const vector<vector<Point>> &rects, vector<int> &labels, double eps) {
	return cv::partition(rects,labels,[eps](const vector<Point> &a, const vector<Point> &b){return norm(a,b)<eps;});
}
void centers(const vector<vector<Point>> &rects, const vector<int> &labels, int C, vector<vector<Point>> &center) {
	center.resize(C,vector<Point>(4));
	vector<int> ct(C,0);
	for (int i=0; i<rects.size(); i++) {
		int id = labels[i];
		CV_Assert(id<C);
		center[id][0] += rects[i][0];
		center[id][1] += rects[i][1];
		center[id][2] += rects[i][2];
		center[id][3] += rects[i][3];
		ct[id] ++;
	}
	for (int i=0; i<center.size(); i++) {
		center[i][0] /= ct[i];
		center[i][1] /= ct[i];
		center[i][2] /= ct[i];
		center[i][3] /= ct[i];
	}
}

// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
	int R=8;
	int Z=120;
	Size S(Z,Z);
	vector<Point2f> dst {
		Point2f(0,0),
		Point2f(Z,0),
		Point2f(Z,Z),
		Point2f(0,Z)
	};

 	vector<int> labels;
	int n = cluster(squares, labels, 108);
	cout << squares.size() << " squares and " << n << " clusters" << endl;
	vector<vector<Point> > center;
	centers(squares,labels,n,center);
    //squares = center;
	cout << "cen " << center.size() << endl;

    vector<Mat> crops;
    for( size_t i = 0; i < center.size(); i++ )
    {
		vector<Point2f> pt = sort_cw(center[i]);
        cout << i << " " << Mat(center[i]).t() << " " << Mat(pt).t() << endl;
        Mat proj = getPerspectiveTransform(pt, dst);
        Mat warped; warpPerspective(image, warped, proj, S,INTER_LINEAR);
    	crops.push_back(warped);
    }
    cout << "crops " << crops.size() << endl;
	Mat crp(0,R*Z,image.type());
	Mat row(Z,0,image.type());
    for( size_t i = 0; i < crops.size(); i++ )
    {
	   	hconcat(row,crops[i],row);
    	if (i%R == (R-1)) {
    		vconcat(crp,row,crp);
    		row.create(Z,0,image.type());
    	}
    }
    imshow(wndname, image);
    if (crp.rows &&  crp.cols)
    	imshow("cropped", crp);
    else if (row.rows && row.cols)
    	imshow("cropped", row);
}


int main(int argc, char** argv)
{
    static const char* names[] = { "books.png", 0 };
  //  help();

    if( argc > 1)
    {
     names[0] =  argv[1];
     names[1] =  "0";
    }

    namedWindow( wndname, 1 );
    vector<vector<Point> > squares;

    for( int i = 0; names[i] != 0; i++ )
    {
        Mat image = imread(names[i], 1);
        if( image.empty() )
        {
            cout << "Couldn't load " << names[i] << endl;
            continue;
        }

        findSquares(image, squares);
        drawSquares(image, squares);

        char c = (char)waitKey();
        if( c == 27 )
            break;
    }

    return 0;
}
