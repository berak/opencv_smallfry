#include <iostream>
#include <opencv2/opencv.hpp>
#include "profile.h"

using namespace cv;
using namespace std;

//
// An Introduction to Independent Component Analysis: InfoMax and FastICA algorithms
//   Dominic Langlois, Sylvain Chartier, and Dominique Gosselin   University of OttawaAn
//   Tutorials in Quantitative Methods for Psychology 2010, Vol. 6(1), p. 31-38.
//

Mat whiten(const Mat &input) {
    PROFILE
    Mat X;
    input.convertTo(X, CV_64F, 1.0/255);

    Scalar m,d;
    meanStdDev(X,m,d);
    X -= m[0];
    X /= d[0];
    return X;
}

Mat tanh(const Mat &y) {
    PROFILE;
    Mat tanh1, tanh2;
    exp(y, tanh1);
    exp(-1 * y, tanh2);
    return (tanh1 - tanh2) / (tanh1 + tanh2);
}

// https://github.com/vislab205/Simo-Lore-FastIca
void fastICA(Mat mix, Mat &w, Mat &output, int snum=0)
{
    PROFILE
    const  int M = mix.rows;    // number of data
    const  int N = mix.cols;    // data dimension
    const int maxIterations = 5000;
    const double epsilon = 0.000001;
    double loss = 0;
    int iters = 0;

    if (M < snum || snum == 0)
    {
        snum = M;
    }

    Mat X = whiten(mix);
    Mat R(snum, N, CV_64FC1);
    randn(R, Scalar(0), Scalar(.5));
    Mat ONE = Mat::ones(M, 1, CV_64FC1);

    for (int i=0; i<snum; ++i)
    {
        PROFILEX("fastica.snum")
        int iteration = 0;
        Mat P = R.row(i).clone(); // [1,N]
        while (iteration <= maxIterations)
        {
            PROFILEX("fastica.iter")
            iteration++, iters++;
            // nonlinearity x^3 and derivative 3*x^2
            Mat P2 = P.clone();
            Mat pxt = P * X.t();
            Mat pxt2 = pxt.mul(pxt);
            Mat pxt3 = pxt2.mul(pxt);
            P = (((3 * pxt2) * ONE) * P - pxt3 * X) / N;  // [3. 13a]

            if (i != 0)
            {
                Mat dp = Mat::zeros(1,N,CV_64FC1);
                for(int j=0; j<i; ++j)
                {
                    Mat wj = R.row(j);
                    dp += P * wj.t() * wj; // [3. 13c]
                }
                P -= dp;
            }
            double Pnorm = norm(P, 4); // [3. 13d]
            P = P / Pnorm;

            loss = min(norm(P - P2, 4), norm(P + P2, 4));
            if (loss < epsilon || iteration == maxIterations)
            {
                P.copyTo(R.row(i));
                break;
            }
        }
    }
    cout << "fastica " << iters << " " << loss << endl;
    w = R * X.t();
    output = R;
}

void ICA(const Mat &mix, Mat &w, Mat &s, int snum=0, int maxIter=5000, double epsilon=1e-9)
{
    PROFILE
    float learn_rate = 0.95;
    int nmix = mix.rows;
    if (snum == 0 || snum > nmix)
        snum = nmix;

    Mat x = whiten(mix);

    Mat I = Mat::eye(snum, snum, CV_64F);
    w.create(snum, nmix, CV_64F);
    randu(w, Scalar(-.01), Scalar(.01));

    double loss=0;
    int iter=0;
    for (; iter<maxIter; iter++)
    {
        // [3.(9)]  w(t+1) = w(t) + lrn * (I - f(Y)*YT) * w(t)
        PROFILEX("ica.iter")
        Mat y = w * x;

        // nonlinearity, use x^3 instead of tanh (much faster)
        Mat fy; pow(y, 3, fy);
        fy = y - fy;  // sub gaussian

        Mat gradient = (I - fy * y.t());

        loss = sum(gradient)[0];
        if (abs(loss) < epsilon)
            break;

        w = w + learn_rate * gradient * w;
    }
    cout << "ica     " << iter << " " << abs(loss) << endl;

    s = w * x;
}

void viz(const Mat &output, String s)
{
    Mat result;
    normalize(output,result,255,0,NORM_MINMAX,CV_8U);
    Mat r1 = result.row(0).reshape(1,100);
    Mat r2 = result.row(1).reshape(1,100);
    Mat r3 = result.row(2).reshape(1,100);
    Mat r;
    hconcat(r1, r2, r);
    hconcat(r, r3, r);
    imshow(s, r);
    waitKey(50);
}

int main()
{
    Size siz(100,100);
    Mat i1 = imread("img/h1.png",0);    resize(i1,i1,siz);
    Mat i2 = imread("img/h2.png",0);    resize(i2,i2,siz);
    Mat i3 = imread("img/h3.png",0);    resize(i3,i3,siz);

    // to seperate 3 inputs, we need 3 mixes
    Mat m1 = i1*.4 + i2*.3 + i3*.3;
    Mat m2 = i1*.3 + i2*.4 + i3*.3;
    Mat m3 = i1*.3 + i2*.3 + i3*.4;

    // visualization
    Mat m; hconcat(m1,m2,m); hconcat(m,m3,m);
    imshow("mix", m);
    Mat o; hconcat(i1,i2,o); hconcat(o,i3,o);
    imshow("orig", o);
    waitKey(10);

    // flatten / stack them to a single input Mat
    Mat mix;
    mix.push_back(m1.reshape(1,1));
    mix.push_back(m2.reshape(1,1));
    mix.push_back(m3.reshape(1,1));

    Mat w,result;
    int64 seed = 0xffffffff;

    // infomax
    theRNG().state = seed;
    ICA(mix, w, result, 3);
    viz(result,"ica");

    // fastica
    theRNG().state = seed;
    fastICA(mix, w, result, 3);
    viz(result,"fast");

    // pca, just for comparison
    PCA pca(mix,Mat(),0,3);
    viz(pca.eigenvectors, "pca");

    waitKey(0);
    return 0;
}
