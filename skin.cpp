//
//Jones, M.J.; Rehg, J.M., "Statistical color models with application to skin detection,"
// Computer Vision and pattern Recognition, 1999. IEEE Computer Society Conference on. , vol.1, no., pp.,280 Vol. 1, 1999
//
// simplified from: https://github.com/Canas/skin_detector

#include "opencv2/opencv.hpp"
using namespace cv;

struct SkinDetector
{
	Mat_<Vec3d> _mean;
	Mat_<Vec3d> _cov;
	Mat_<double> _covDet;
	Mat_<double> _weight;
	double threshold;

	SkinDetector(double threshold)
		: _mean(2,16)
		, _cov(2,16)
		, _covDet(2,16)
		, _weight(2,16)
		, threshold(threshold)
	{
		Mat_<double> m(2,16*3); m <<
			// skin
			17.76, 29.94, 73.53,
			217.49, 233.94, 249.71,
			96.95, 116.25, 161.68,
			114.40, 132.62, 186.07,
			51.58, 98.37, 189.26,
			90.84, 152.20, 247.00,
			37.76, 72.66, 150.10,
			156.34, 171.09, 206.85,
			120.04, 152.82, 212.78,
			138.94, 175.43, 234.87,
			74.89, 97.74, 151.19,
			59.82, 77.55, 120.52,
			82.32, 119.62, 192.20,
			87.24, 136.08, 214.29,
			38.06, 54.33, 99.57,
			176.91, 203.08, 238.88,
			// non-skin
			253.82, 254.41, 254.37,
			8.52, 8.09, 9.39,
			96.57, 96.95, 91.53,
			159.06, 162.49, 160.44,
			46.33, 63.23, 74.98,
			18.31, 60.88, 121.83,
			91.04, 154.88, 202.18,
			206.55, 201.93, 193.06,
			61.55, 57.14, 51.88,
			25.32, 26.84, 30.88,
			131.95, 85.96, 44.97,
			230.70, 236.27, 236.02,
			164.12, 191.20, 207.86,
			188.17, 148.11, 99.83,
			123.10, 131.92, 135.06,
			66.88, 103.89, 135.96;
		_mean = m.reshape(3,2);

		Mat_<double> c(2,16*3); c <<
			// skin
			112.80, 121.44, 765.40,
			396.05, 154.44, 39.94,
			162.85, 60.48, 291.03,
			198.27, 64.60, 274.95,
			250.69, 222.40, 633.18,
			609.92, 691.53, 65.23,
			257.57, 200.77, 408.63,
			572.79, 155.08, 530.08,
			243.90, 84.52, 160.57,
			279.22, 121.57, 163.80,
			175.11, 73.56, 425.40,
			151.82, 70.34, 330.45,
			259.15, 92.14, 152.76,
			270.19, 140.17, 204.90,
			151.29, 90.18, 448.13,
			404.99, 156.27, 178.38,
			// non-skin
			5.46, 2.81, 2.77,
			32.48, 33.59, 46.84,
			436.58, 156.79, 280.69,
			591.24, 115.86, 355.98,
			361.27, 245.95, 414.84,
			237.18, 1383.53, 2502.24,
			1582.52, 1766.94, 957.42,
			447.28, 190.23, 562.88,
			433.40, 191.77, 344.11,
			182.41, 118.65, 222.07,
			963.67, 840.52, 651.32,
			331.95, 117.29, 225.03,
			533.52, 237.69, 494.04,
			916.70, 654.95, 955.88,
			388.43, 130.30, 350.35,
			350.36, 642.20, 806.44;
		_cov = c.reshape(3,2);

		// precompute sqrt(det(cov)):
		static double pow2pi = pow(2 * CV_PI, 1.5);
		for (int i=0; i<16; i++)
		{
			_covDet(0,i) = pow2pi * pow(_cov(0,i)[0] * _cov(0,i)[1] * _cov(0,i)[2], 0.5);
			_covDet(1,i) = pow2pi * pow(_cov(1,i)[0] * _cov(1,i)[1] * _cov(1,i)[2], 0.5);
		}

		_weight <<
			// skin
			0.0294,0.0331,0.0654,0.0756,0.0554,0.0314,0.0454,0.0469,0.0956,0.0763,0.1100,0.0676,0.0755,0.0500,0.0667,0.0749,
			// non-skin
			0.0637,0.0516,0.0864,0.0636,0.0747,0.0365,0.0349,0.0649,0.0656,0.1189,0.0362,0.0849,0.0368,0.0389,0.0943,0.0477;
	}

	double gaussianMixture(int id, const Vec3d &RGBpixel)
	{
		double P = 0;
		for (int i=0; i<16; i++)
		{
			Vec3d a = RGBpixel - _mean(id, i);
			Vec3d cov = _cov(id, i);
			double C = _covDet(id,i);
			double D = -0.5 * ((a[0]*a[0]) / cov[0] + (a[1]*a[1]) / cov[1] + (a[2]*a[2]) / cov[2]);
			double w = _weight(id, i);
			P = P + (w/C) * exp(D);
		}
		return P;
	}

	bool detect(const Vec3d &pixel)
	{
		double P_skin   = gaussianMixture(0, pixel);
		double P_noskin = gaussianMixture(1, pixel);
		// bayes
		double rate = P_skin / P_noskin;
		return (rate > threshold);
	}
};

//
// ----snip-test---8<------------------------------------------
//

#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
	Mat img = imread(argv[1]);
	Mat out(img.size(), img.type());
	SkinDetector skin(1.2); // threshold

	int64 t0 = getTickCount();
	for (int i=0; i<img.rows; i++)
	{
		Vec3b * pp = img.ptr<Vec3b>(i);
		for (int j=0; j<img.cols; j++)
		{
			Vec3b p = pp[j];
			Vec3d pd(p);
			bool s = skin.detect(pd);
			out.at<Vec3b>(i,j) = s ? p : Vec3b(255,255,255);
		}
	}
	int64 t1 = getTickCount();
	cerr << "ticks: " << (t1-t0) << " " << (t1-t0)/getTickFrequency() << endl;
	imshow("in", img);
	imshow("out", out);
	waitKey();
	return 0;
}
