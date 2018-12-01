#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace std;
using namespace cv;


const int GRID=1;
const int NTESTS=1000;


double dt(int64 t) { return double(t)/getTickFrequency(); }

struct Profile
{
    string name;
    int64 t; // accumulated time
    int64 c; // accumulated function calls

    Profile(const string & name)
        : name(name)
        , t(0)
        , c(0)
    {}

    ~Profile()
    {
        cerr << format("%-24s %8u ",name.c_str(),c);
        cerr << format("%13.6f ",dt(t/c));
        cerr << format("%13.6f ",dt(t));
        cerr << format("%14u",t);
        cerr << endl;
    }

    struct Scope
    {
	    Profile & p;
        int64 t;

	    Scope(Profile & p)
		    : p(p)
            //, t(get_ticks())
            , t(getTickCount())
        {}

	    ~Scope()
	    {
            int64 t1 = getTickCount();
            //int64 t1 = get_ticks();
            if ( t1 > t )
            {
                p.t += t1 - t;
                p.c ++;
            }
	    }
    };
};

#define PROFILEX(s) static Profile _a_rose(s); Profile::Scope _is_a_rose_is(_a_rose);
#define PROFILE PROFILEX(__FUNCTION__)



// histogram calculation seems to be the bottleneck.
void calc_hist(const Mat_<uchar> & feature, Mat_<float> & hist, int histSize)
{   PROFILE;
    for ( int i=0; i<feature.rows; i++ )
    {
        for ( int j=0; j<feature.cols; j++ )
        {
            uchar bin = int(feature(i,j)) * histSize / 256;
            hist( bin ) += 1.0f;
        }
    }
}

void hist(const Mat & feature, Mat & hist, int histSize=256, int gridX=1, int gridY=1)
{   PROFILE;
    hist.release();
    const float range[] = { 0, 256 } ;
    const float* histRange = { range };
    int sw = (feature.cols)/(gridX+1);
    int sh = (feature.rows)/(gridY+1);
    for ( int i=0; i<gridX; i++ )
    {
        for ( int j=0; j<gridY; j++ )
        {
            Rect patch(i*sw,j*sh,sw,sh);
            Mat fi( feature, patch );
            Mat_<float> h(1,histSize,0.0f);
            // calcHist( &fi, 1, 0, Mat(), h, 1, &histSize, &histRange, true, false );
            calc_hist(fi,h,histSize);
            hist.push_back(h.reshape(1,1));
        }
    }
    normalize(hist.reshape(1,1),hist);
}


//
// reference impl, walking over pixels.
//
struct lbp_pix
{
    void operator()( const Mat &z, Mat & h ) const
    {   PROFILEX("lbp_pix");
        h  = Mat::zeros(1,256,CV_32F);
        Mat_<uchar> fI(z.size());
        Mat_<uchar> img(z);
        const int m=1;
        for ( int r=m; r<z.rows-m; r++ )
        {
            for ( int c=m; c<z.cols-m; c++ )
            {
                uchar v = 0;
                uchar cen = img(r,c);
                v |= (img(r-1,c  ) > cen) << 0;
                v |= (img(r-1,c+1) > cen) << 1;
                v |= (img(r  ,c+1) > cen) << 2;
                v |= (img(r+1,c+1) > cen) << 3;
                v |= (img(r+1,c  ) > cen) << 4;
                v |= (img(r+1,c-1) > cen) << 5;
                v |= (img(r  ,c-1) > cen) << 6;
                v |= (img(r-1,c-1) > cen) << 7;
                fI(r,c) = v;
            }
        }
        hist(fI,h,256,GRID,GRID);
    }
};

//
// here be matlab dragons.
//
//  trying to adapt Antonio Fernandez's hep.m, some caveats and pitfalls:
//  * since a Matop in opencv like A>B results in a Mat filled with [0xff or 0], (not [0 or 1]),
//    we have to use & and | instead of * and +
//
//

#define SHIFTED_MATS_3x3 \
        int M = I.rows; \
        int N = I.cols; \
        Mat I7 = I(Range(1,M-2), Range(1,N-2));\
        Mat I6 = I(Range(1,M-2), Range(2,N-1));\
        Mat I5 = I(Range(1,M-2), Range(3,N  ));\
        Mat I4 = I(Range(2,M-1), Range(3,N  ));\
        Mat I3 = I(Range(3,M  ), Range(3,N  ));\
        Mat I2 = I(Range(3,M  ), Range(2,N-1));\
        Mat I1 = I(Range(3,M  ), Range(1,N-2));\
        Mat I0 = I(Range(2,M-1), Range(1,N-2));\
        Mat Ic = I(Range(2,M-1), Range(2,N-1));\


struct LBP_3x3
{
    void operator()( const Mat &I, Mat & h ) const
    {   PROFILEX("LBP");

        SHIFTED_MATS_3x3;

        Mat fI = ((I7>Ic)&128) |
             ((I6>Ic)&64)  |
             ((I5>Ic)&32)  |
             ((I4>Ic)&16)  |
             ((I3>Ic)&8)   |
             ((I2>Ic)&4)   |
             ((I1>Ic)&2)   |
             ((I0>Ic)&1);

        hist(fI,h,256,GRID,GRID);
    }
};

struct BGC1_3x3
{
    void operator()( const Mat &I, Mat & h ) const
    {   PROFILEX("BGC1");

        SHIFTED_MATS_3x3;

        Mat fI = ((I7>=I0)&128) |
             ((I6>=I7)& 64) |
             ((I5>=I6)& 32) |
             ((I4>=I5)& 16) |
             ((I3>=I4)&  8) |
             ((I2>=I3)&  4) |
             ((I1>=I2)&  2) |
             ((I0>=I1)&  1);

        hist(fI,h,256,GRID,GRID);
    }
};

struct RT_3x3
{
    void operator()( const Mat &I, Mat & h ) const
    {   PROFILEX("RT");

        SHIFTED_MATS_3x3;

        Mat fI = (Ic>I7) | (Ic>I6) | (Ic>I5) | (Ic>I4) | (Ic>I3) | (Ic>I2) | (Ic>I1) | (Ic>I0);

        hist(fI,h,9,GRID,GRID);
    }
};

struct LTP_3x3
{
    int kerP1;

    LTP_3x3(int k=5) : kerP1(k) {}

    void operator()( const Mat &I, Mat & h ) const
    {   PROFILEX("LTP");

        SHIFTED_MATS_3x3;

        Mat Icplus  = Ic+kerP1;
        Mat Icminus = Ic-kerP1;
        Mat fUp = ((I7>=Icplus)&128) |
                  ((I6>=Icplus)& 64) |
                  ((I5>=Icplus)& 32) |
                  ((I4>=Icplus)& 16) |
                  ((I3>=Icplus)&  8) |
                  ((I2>=Icplus)&  4) |
                  ((I1>=Icplus)&  2) |
                  ((I0>=Icplus)&  1);
        Mat fDn = ((I7<=Icminus)&128) |
                  ((I6<=Icminus)& 64) |
                  ((I5<=Icminus)& 32) |
                  ((I4<=Icminus)& 16) |
                  ((I3<=Icminus)&  8) |
                  ((I2<=Icminus)&  4) |
                  ((I1<=Icminus)&  2) |
                  ((I0<=Icminus)&  1);
        Mat h1,h2;
        hist(fUp,h1,256,GRID,GRID);
        hist(fDn,h2,256,GRID,GRID);
        h.push_back(h1);
        h.push_back(h2);
        h = h.reshape(1,1);
    }
};



struct  LQP_3x3
{
    int kerP1;
    int kerP2;

    LQP_3x3(int k1=5,int k2=5) : kerP1(k1), kerP2(k2) {}

    void operator()( const Mat &I, Mat & h ) const
    {   PROFILEX("LQP");
        Mat fI_2,fI_1,fI1,fI2;

        SHIFTED_MATS_3x3;

        Mat Icplus1  = Ic+kerP1;
        Mat Icplus2  = Ic+kerP2;
        Mat Icminus1 = Ic-kerP1;
        Mat Icminus2 = Ic-kerP2;
        fI_2 =  ((I7<Icminus2)&128 ) |
                ((I6<Icminus2)& 64 ) |
                ((I5<Icminus2)& 32 ) |
                ((I4<Icminus2)& 16 ) |
                ((I3<Icminus2)&  8 ) |
                ((I2<Icminus2)&  4 ) |
                ((I1<Icminus2)&  2 ) |
                ((I0<Icminus2)&  1 );
        fI_1 =  (((I7>=Icminus2) &(I7<Icminus1))&128 ) |
                (((I6>=Icminus2) &(I6<Icminus1))& 64 ) |
                (((I5>=Icminus2) &(I5<Icminus1))& 32 ) |
                (((I4>=Icminus2) &(I4<Icminus1))& 16 ) |
                (((I3>=Icminus2) &(I3<Icminus1))&  8 ) |
                (((I2>=Icminus2) &(I2<Icminus1))&  4 ) |
                (((I1>=Icminus2) &(I1<Icminus1))&  2 ) |
                (((I0>=Icminus2) &(I0<Icminus1))&  1 );
        fI1 =   (((I7>=Icplus1) &(I7<Icplus2))&128 ) |
                (((I6>=Icplus1) &(I6<Icplus2))& 64 ) |
                (((I5>=Icplus1) &(I5<Icplus2))& 32 ) |
                (((I4>=Icplus1) &(I4<Icplus2))& 16 ) |
                (((I3>=Icplus1) &(I3<Icplus2))&  8 ) |
                (((I2>=Icplus1) &(I2<Icplus2))&  4 ) |
                (((I1>=Icplus1) &(I1<Icplus2))&  2 ) |
                (((I0>=Icplus1) &(I0<Icplus2))&  1 );
        fI2 =   ((I7>=Icplus2)&128 ) |
                ((I6>=Icplus2)& 64 ) |
                ((I5>=Icplus2)& 32 ) |
                ((I4>=Icplus2)& 16 ) |
                ((I3>=Icplus2)&  8 ) |
                ((I2>=Icplus2)&  4 ) |
                ((I1>=Icplus2)&  2 ) |
                ((I0>=Icplus2)&  1 );

        Mat h1,h2,h3,h4;
        hist(fI_2,h1,256,GRID,GRID);
        hist(fI_1,h2,256,GRID,GRID);
        hist(fI1, h3,256,GRID,GRID);
        hist(fI2, h4,256,GRID,GRID);
        h.push_back(h1);
        h.push_back(h2);
        h.push_back(h3);
        h.push_back(h4);
        h = h.reshape(1,1);
    }
};


template<class HEP_3x3>
void bench(const Mat &m, const HEP_3x3 &krn, const string &fn)
{
    uint64 t0 = getTickCount();
    Mat h;
    for ( int i=0; i<NTESTS; i++ )
    {
        h.release();
        krn(m,h);
    }
    uint64 t1 = getTickCount();
    cerr << format("%-10s %5.4f %7d", fn.c_str(), (t1-t0)/getTickFrequency(), h.cols) << endl;
}

int main()
{   PROFILE;
    Mat m=imread("../demo/img/lena1.png",0);
    resize(m,m,Size(120,120));
    cerr << GRID << " " << m.size() << endl;

    bench(m,lbp_pix(),"lbp1");
    bench(m,LBP_3x3(),"lbp2");
    bench(m,LTP_3x3(4),"LTP");
    bench(m,LQP_3x3(4,4),"LQP");
    bench(m,RT_3x3(),"RT");
    bench(m,BGC1_3x3(),"BGC1");

    cerr << "." << endl;
    return 0;
}
