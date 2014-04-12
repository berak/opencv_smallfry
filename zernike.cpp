#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include <cstdio>
#include <iostream>
using namespace cv;
using namespace std;


//
// this code tries to apply zernike moments to a (patched) image as a 'textron' similarity measure
//


//
// nice explanatory snippet taken from: zernike_radial_polynomials.m
// we will be using the precalculated n m Zernike polynomials below.
//

//
//    Written by Mohammed S. Al-Rawi, 2007-2008
//    Last updated 2011.
//    rawi707@yahoo.com
// 
//  The fast method presented in this code:
//   The method implemented in this work is q-recurseive see Ref.
//   Way, C.-C., Paramesran, R., and Mukandan, R., A comparative analysis of algorithms for fast computation of Zernike moments, Pattern Recognition 36 (2003) 731-742.
//   It uses radial polynomials of fixed order p with a varying index q to
//   compute Zernike moments
//
//
//  What are Zernike polynomials?
//    The radial Zernike polynomials are the radial portion of the
//    Zernike functions, which are an orthogonal basis on the unit
//    circle.  The series representation of the radial Zernike
//    polynomials is
// 
//           (n-m)/2
//             __
//     m      \       s                                          n-2s
//    Z(r) =  /__ (-1)  [(n-s)!/(s!((n-m)/2-s)!((n+m)/2-s)!)] * r
//     n      s=0
// 
//    The following table shows the first 12 polynomials.
// 
//        n    m    Zernike polynomial    Normalization
//        ---------------------------------------------
//        0    0    1                        sqrt(2)
//        1    1    r                           2
//        2    0    2*r^2 - 1                sqrt(6)
//        2    2    r^2                      sqrt(6)
//        3    1    3*r^3 - 2*r              sqrt(8)
//        3    3    r^3                      sqrt(8)
//        4    0    6*r^4 - 6*r^2 + 1        sqrt(10)
//        4    2    4*r^4 - 3*r^2            sqrt(10)
//        4    4    r^4                      sqrt(10)
//        5    1    10*r^5 - 12*r^3 + 3*r    sqrt(12)
//        5    3    5*r^5 - 4*r^3            sqrt(12)
//        5    5    r^5                      sqrt(12)
//        ---------------------------------------------
// 


//
// to calculate the ZernikeMoment of a given patch image, 
// this implementation (loosely) follows the pseudocode example in figure 3 of
//   "Anovel approach to the fast computation of Zernikemoments" [Sun-Kyoo Hwang,Whoi-Yul Kim] 2006
//
// since the radial zernike polynomials as well as rho and theta 
//   are 'constant' for a given patchsize (or say, independant of our image),
//   we can cache a Mat with the (radial*cos(m*theta)) term for each of our 10 or so moments,
//   so calculating the ZernikeMoment of a patch (in a later stage) 
//   boils down to a NxN matrix-mult, and a sum over that
//
// omitting the 1st 2 polynomials above(since they don't add much gain),
//   so this has 10 moments
//

class Zernike
{
    //;) don't bother optimizing below private code (at all), this is used to generate lut's, once per Zernike instance 
    static double radpol_11(double r) { return (r)/(2.0); }
    static double radpol_20(double r) { return (2*r*r - 1)/sqrt(2.0); }
    static double radpol_22(double r) { return (r*r)/sqrt(6.0); }
    static double radpol_31(double r) { return (3*r*r*r - 2*r*r)/sqrt(8.0); }
    static double radpol_33(double r) { return (r*r*r)/sqrt(8.0); }
    static double radpol_40(double r) { return (6*r*r*r*r - 6*r*r+1)/sqrt(10.0); }
    static double radpol_42(double r) { return (4*r*r*r*r - 3*r*r)/sqrt(10.0); }
    static double radpol_44(double r) { return (r*r*r*r)/sqrt(10.0); }
    static double radpol_51(double r) { return (10*r*r*r*r - 12*r*r*r + 3*r)/sqrt(12.0); }
    static double radpol_53(double r) { return (5*r*r*r*r - 4*r*r*r)/sqrt(12.0); }
    static double radpol_55(double r) { return (r*r*r*r*r)/sqrt(12.0); }

    //! we only save the real/cos part of the (originally complex) equation here.
    //
    //  please prove me wrong, but the base assumption here is, that this is a hilbert space
    //      [since sin(m*t) is just a 90° shifted version of cos(m*t)] ,
    //    so, we're safe to skip the (dependant) sin/imaginary component.
    //
    void mat_cos(Mat & zm, int N, double maumau, double(*radicalchic)(double) )
    {
        zm = Mat::zeros(N,N,CV_32F);
        int cnt = 0;
        for ( int i=0; i<N; i++ ) 
        {
            for ( int j=0; j<N; j++ ) 
            {
                double a(2*i-N+1);
                double b(N-1-j*2);
                double rho = sqrt(a*a + b*b);
                double theta = atan(b/a);
                double radial = radicalchic(rho);
                zm.at<float>(i,j) = float(radial * cos(maumau * theta));
            }
        }
        zm /= (N*N); // normalized [-1,1]
    }

    Mat zerm[10];    // 10 max features
    int N;           // patchsize
    int nfeatures;   // you might want to use less than 10 features

public:


    //
    //! precalculate the (radial*cos(m*theta)) term for each of our 10 moments
    //! resultant featuresize will be nfeatures*N*N
    //
    Zernike(int n=8, int used=10)
        : N(n)
        , nfeatures(min(used, 10))
    {
        mat_cos(zerm[0], N, 0.0, radpol_20);
        mat_cos(zerm[1], N, 2.0, radpol_22);
        mat_cos(zerm[2], N, 1.0, radpol_31);
        mat_cos(zerm[3], N, 3.0, radpol_33);
        mat_cos(zerm[4], N, 0.0, radpol_40);
        mat_cos(zerm[5], N, 2.0, radpol_42);
        mat_cos(zerm[6], N, 4.0, radpol_44);
        mat_cos(zerm[7], N, 1.0, radpol_51);
        mat_cos(zerm[8], N, 3.0, radpol_53);
        mat_cos(zerm[9], N, 5.0, radpol_55);
    }


    //
    //! adds nfeatures elems for a NxN patch to a feature Mat
    //! expects single channel float Mats as input
    //
    void compute_patch( const Mat & patch, Mat & features )
    {
        for (int i=0; i<nfeatures; i++) 
        {
            Mat c;
            multiply(patch, zerm[i], c); // per element
            features.push_back(float(sum(c)[0]));
        }
    }

    //
    //! calculates an nfeatures*N*N feature vec per image, 
    //!  the (L2)norm of it will be our distance metrics for comparing images.
    //
    void compute( const Mat & img, Mat & features )
    {       
        Mat m;
        if ( img.type() != CV_32F )
            img.convertTo(m, CV_32F);
        else
            m=img;

        // the trick with the precalculated (radial*cos(m*theta)) term requires a fixed patch size,
        // so let's try to 'equalize' differently sized images here
        //   downside: this puts a penalty on (small) input images < NxN ,
        //      please let me know, if you find something better here.
        cv::resize(m, m, Size(N*N, N*N));

        for (int i=0; i<N; i++) 
        {
            for (int j=0; j<N; j++) 
            {
                Mat patch = m(Rect(j*N, i*N, N, N));
                compute_patch(patch, features);
            }
        }
    }
};

int main()
{
    // some stupid example image from 'funnelled lfw faces in the wild'
    Mat m = imread("Adrien_Brody_0004.jpg",0);

    // initialize our structures
    int64 t0 = getTickCount();
    int patches  = 24;
    int features = 10;
    Zernike zerm(patches,features);
    int64 t1 = getTickCount();

    // get a concatenated feature vector of patches*patches*nfeatures for the image
    Mat feat;
    zerm.compute(m,feat);
    int64 t2 = getTickCount();
    
    // just to simulate real life usage, 
    // - test, how long 10000 comparisons of those would take:
    for ( int i=0; i<10000; i++ ) norm(feat,feat,NORM_L2);
    int64 t3 = getTickCount();
    for ( int i=0; i<10000; i++ ) norm(feat,feat,NORM_INF);
    int64 t4 = getTickCount();

    cerr << "time_init: " << (double(t1-t0)/getTickFrequency()) << endl;;
    cerr << "time_feat: " << (double(t2-t1)/getTickFrequency()) << endl;;
    cerr << "time_l2  : " << (double(t3-t2)/getTickFrequency()) << endl;;
    cerr << "time_inf : " << (double(t4-t3)/getTickFrequency()) << endl;;
    cerr << "image    : " << m.rows << " x " << m.cols << endl;
    cerr << "patches  : " << patches << " x " << features << endl;
    cerr << "features : " << feat.rows << " x " << feat.cols << endl;
    imshow("mmm",m);
    waitKey();
    return 0;
}
