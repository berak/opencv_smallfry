#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
using namespace cv;
#include <fstream>
using namespace std;

//
// https://sites.google.com/site/simpmatrix/
//

template<class T>
double sign(T v)
{
	if(v>0) return 1.0;
	else if(v<0) return -1.0;
	else return 0.0;
}


// Multidimensional scaling (MDS)
// This function re-implements Laurens van der Maaten's MDS in his Matlab Toolbox for Dimensionality Reduction
// The Matlab MDS can be downloaded at http://crcv.ucf.edu/source/dimension
Mat_<double> MDS_UCF(const Mat_<double> &D, int dim, int iter)
{
	if(D.rows != D.cols)
	{
		printf("Input distance matrix to MDS is not square.\n");
		exit(1);
	}
	if(dim<1)
	{
		printf("Invalid dimension for MDS.\n");
		exit(1);
	}
	if(iter<1)
	{
		printf("Invalid number of iterations for MDS.\n");
		exit(1);
	}

	Mat_<double>  X(D.cols, dim);
	randu(D,0,1.0);
	double D_mean = mean(D)[0]; // mean value of distance matrix
	D += (-0.5); // move to the center
	X *= (0.1*D_mean/(1.0/3.0*sqrt((double)dim))); // before this step, mean distance is 1/3*sqrt(d)


	double lr=0.05; // learning rate
	double r=2; // metric
	int n=D.rows; // number of vectors


	Mat_<double> dh(n,n,0.0);
	Mat_<double> pmat(n,dim);
	Mat_<double> dhdum(n,1);
	Mat_<double> dhmat(n-1,dim,0.0);

	Mat_<int> RP(n,iter); // the matrix for random permutation numbers
    randu(RP, 0, n);
cerr << "RP" << RP << endl;
	int i,j;
	double temp;
	int m;

	printf("MDS iteration:");
	for(int it=0;it<iter;it++) // iterations
	{
		if(it%10==0) printf("\n");
		printf("%3d  ",it+1);
		for(int rp=0;rp<n;rp++) // work on each vector in a randomly permuted order
		{
			m=RP(rp,it);

			for(i=0;i<n;i++)
			{
				for(j=0;j<dim;j++)
				{
					pmat(i,j) = X(m,j) - X(i,j);
				}
			}

			for(i=0;i<n;i++)
			{
				temp=0;
				for(j=0;j<dim;j++)
				{
					temp+= pow(fabs(pmat(i,j)), r);
				}
				dhdum(i,0) = pow(temp,1/r);
			}

			for(i=0;i<n;i++)
			{
				if(i==m) continue;

				dh(m,i) = dhdum(i,0);
				dh(i,m) = dhdum(i,0);
			}

			for(i=0;i<n-1;i++)
			{
				int ii=i;
				if(i>=m) ii=i+1;
				temp = lr * ( dhdum(ii,0) - D(ii,m) ) * pow( dhdum(ii,0), 1-r );
				for(j=0;j<dim;j++)
				{
					dhmat(i,j) = temp;
				}
			}

			for(i=0;i<n-1;i++)
			{
				int ii=i;
				if(i>=m) ii=i+1;
				for(j=0;j<dim;j++)
				{
					temp = X(ii,j);
					temp+= dhmat(i,j) * pow(fabs( pmat(ii,j) ),r-1) * sign<double>(pmat(ii,j));

					X(ii,j) = temp;
				}
			}
		}
	}
	cerr << X << endl;
	printf("\n");

	return X;
}



int main(int argc, const char * argv[])
{
	Mat_<double> dist(5,5);
	randu(dist, 0, 1.0);
	dist.diag() = 0;
	cerr << dist << endl;
	Mat X = MDS_UCF(dist,3,1);
	cerr << X << endl;

	/*Mat img = imread("shadow2.png");
	Mat ycbcr; cvtColor(img, ycbcr, COLOR_BGR2YCrCb);
	//Mat ycbcr; cvtColor(img, ycbcr, COLOR_BGR2YCrCb);
	Scalar m,d;
	Mat chn[3];
	split(ycbcr, chn);
	Mat Y;
	equalizeHist(chn[0], Y);
	meanStdDev(Y, m, d);
	Mat shadow = Y < d[0];
	Scalar m2,d2;
	meanStdDev(Y, m2, d2, Y > d[0]);
	Scalar m3,d3;
	meanStdDev(Y, m3, d3, Y < d[0]);
	//shadow = Y < d2[0];
	cerr << m << d << endl;
	cerr << m2 << d2 << endl;
	cerr << m3 << d3 << endl;
	imshow("org", img);
	imshow("sha", shadow);
	imshow("mask", Y > d[0]);
	waitKey();
    */
    return 0;
}
