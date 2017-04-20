#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

inline int sign(double v)
{
    if (v > 0) return 1;
    if (v < 0) return -1;
    return 0;
}

double cv_matchShapes(InputArray contour1, InputArray contour2, int method, double)
{
    double ma[7], mb[7];
    int i, sma, smb;
    double eps = 1.e-5;
    double mmm;
    double result = 0;
    bool anyA = false, anyB = false;

    HuMoments( moments(contour1), ma );
    HuMoments( moments(contour2), mb );

// debug code:
Mat Ma(1,7,CV_64F, ma);
Mat Mb(1,7,CV_64F, mb);
cerr << Ma << endl<< Mb << endl;

    switch (method)
    {
    case 1:
        for( i = 0; i < 7; i++ )
        {
            double ama = fabs( ma[i] );
            double amb = fabs( mb[i] );

            if (ama > 0)
                anyA = true;
            if (amb > 0)
                anyB = true;

            if ( ama > eps && amb > eps )
            {
                sma = sign(ma[i]);
                smb = sign(mb[i]);

                ama = 1. / (sma * log10( ama ));
                amb = 1. / (smb * log10( amb ));
                result += fabs( -ama + amb );
            }
        }
        break;

    case 2:
        for( i = 0; i < 7; i++ )
        {
            double ama = fabs( ma[i] );
            double amb = fabs( mb[i] );

            if (ama > 0)
                anyA = true;
            if (amb > 0)
                anyB = true;

            if ( ama > eps && amb > eps )
            {
                sma = sign(ma[i]);
                smb = sign(mb[i]);

                ama = sma * log10( ama );
                amb = smb * log10( amb );
                result += fabs( -ama + amb );
            }
        }
        break;

    case 3:
        for( i = 0; i < 7; i++ )
        {
            double ama = fabs( ma[i] );
            double amb = fabs( mb[i] );

            if (ama > 0)
                anyA = true;
            if (amb > 0)
                anyB = true;

            if ( ama > eps && amb > eps )
            {
                sma = sign(ma[i]);
                smb = sign(mb[i]);

                ama = sma * log10( ama );
                amb = smb * log10( amb );
                mmm = fabs( (ama - amb) / ama );
                if( result < mmm )
                    result = mmm;
            }
        }
        break;
    default:
        CV_Error( CV_StsBadArg, "Unknown comparison method" );
    }

    //If anyA and anyB are both true, the result is correct.
    //If anyA and anyB are both false, the distance is 0, perfect match.
    //If only one is true, then it's a false 0 and return large error.

    //
    // just comment it out, and see, what happens !
    //
    if (anyA != anyB)
        result = DBL_MAX;

    return result;
}


int main(int argc, char **argv)
{
    Mat img(100,100,CV_8U,Scalar(0));
    Mat img1(100,100,CV_8U,Scalar(0));
    rectangle(img,Rect(30,30,60,20),Scalar(255),-1);
    circle(img1,Point(60,60),10,Scalar(255),-1);
    double d = cv_matchShapes(img,img1,1,0);
    cerr << d << endl;
    //imshow("I",img);
    //waitKey();
    return 0;
}
