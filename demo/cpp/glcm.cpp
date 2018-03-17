
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;


typedef double (*Feature)(int i, int j, const Mat_<float> &gl);

double energy(int i, int j, const Mat_<float> &gl)
{
    return gl(i,j) * gl(i,j);
}
double contrast(int i, int j, const Mat_<float> &gl)
{
    return (i - j)*(i - j)*gl(i, j);
}
double homogenity(int i, int j, const Mat_<float> &gl)
{
    if (gl(i, j) != 0)
        return - gl(i, j)*log10(gl(i, j));
    return 0;
}
double entropy(int i, int j, const Mat_<float> &gl)
{
    return gl(i, j) / (1 + abs(i - j));
}

Mat glcm(const Mat_<uchar> &img, int scale)
{
    Mat_<float> gl(256>>scale, 256>>scale, 0.0f);

    //creating glcm matrix with (256 >> scale) levels,
    //  radius=1 and in the horizontal direction
    for (int i = 0; i<img.rows; i++)
    {
        for (int j = 0; j<img.cols - 1; j++)
        {
            int r = img(i, j)     >> scale;
            int c = img(i, j + 1) >> scale;
            gl(r, c) += 1;
        }
    }
    // normalizing glcm matrix for parameter determination
    gl = gl + gl.t();
    gl = gl / sum(gl)[0];
    return gl;
}

double per_patch(const Mat &img, int scale, Feature feature)
{
    int N = 256 >> scale;
    Mat_<float> gl = glcm(img, scale); // 2 ^ 3 -> 256/8=32 bins
    double v = 0;
    for (int i = 0; i<N; i++)
    {
        for (int j = 0; j<N; j++)
        {
            v += feature(i, j, gl);
        }
    }
    return v;
}

Mat glcm_feature(const Mat &img, int W, int scale, Feature feature)
{
    Mat_<float> res(img.size());
    res = 0;
    for (int i=0; i<img.rows-W-1; i++)
    {
        for (int j=0; j<img.cols-W-1; j++)
        {
            Mat patch(img, Rect(j, i, W, W));
            res(i, j) = per_patch(patch, scale, feature);
        }
    }
    return res;
}

int main(int argc, char** argv)
{
    Mat_<uchar> img = imread(argv[1], IMREAD_GRAYSCALE);

    if (img.empty())
    {
        cout << "can not load " << argv[1] << endl;
        return 1;
    }
    imshow("Image", img);

    int W = 5; // window size
    int scale = 4; // 16 bins
    Mat en = glcm_feature(img, W,scale, energy);
    imshow("energy", en);
    Mat co = glcm_feature(img, W,scale, contrast);
    imshow("contrast", co);
    waitKey(0);
    return 0;
}
