#include <opencv2/opencv.hpp>
using namespace cv;

#include <fstream>
using namespace std;

//
// an attempt to simulate "mapped binary patterns" (a 3d embedding of 8 bit lbp space)
// https://gilscvblog.com/2017/01/31/emotion-recognition-in-the-wild-via-convolutional-neural-networks-and-mapped-binary-patterns/
// with opencv means
//

double sign(double v) {
	if(v>0) return 1.0;
	if(v<0) return -1.0;
	return 0.0;
}


//
// https://sites.google.com/site/simpmatrix/
//
// Multidimensional scaling (MDS)
// This function re-implements Laurens van der Maaten's MDS in his Matlab Toolbox for Dimensionality Reduction
// The Matlab MDS can be downloaded at http://crcv.ucf.edu/source/dimension
Mat_<double> MDS_UCF(const Mat_<double> &D, int dim, int iter) {
	CV_Assert(D.rows == D.cols);
	CV_Assert(dim >= 1);
	CV_Assert(iter >= 1);

	Mat_<double> X(D.cols, dim);
	randu(X, 0, 1.0);
	double D_mean = mean(D)[0]; // mean value of distance matrix
	D += (-0.5); // move to the center
	X *= (0.1*D_mean/(1.0/3.0*sqrt((double)dim))); // before this step, mean distance is 1/3*sqrt(d)

	double lr=0.05; // learning rate
	double r=2;     // metric
	int n=D.rows;   // number of vectors

	//Mat_<double> dh(n,n,0.0);
	Mat_<double> pmat(n,dim,0.0);
	Mat_<double> dhdum(n,1,0.0);
	Mat_<double> dhmat(n-1,dim,0.0);

	for(int it=0; it<iter; it++) {
		Mat_<int> RP(1, n); // the matrix for random permutation numbers
	    for (int i=0; i<n; i++) {
	    	RP(i) = i;
	    }
	    randShuffle(RP);

		for(int rp=0; rp<n; rp++) {
			int m = RP(rp); // work on each vector in a randomly permuted order
			for(int i=0; i<n; i++) {
				for(int j=0; j<dim; j++) {
					pmat(i,j) = X(m,j) - X(i,j);
				}
			}

			for(int i=0; i<n; i++) {
				double temp=0;
				for(int j=0; j<dim; j++) {
					temp += pow(fabs(pmat(i,j)), r);
				}
				dhdum(i,0) = pow(temp,1/r);
			}

			/*for(int i=0;i<n;i++) // this is already in http://crcv.ucf.edu/source/dimension%20reduction/mds.m (never used ???)
			{
				if(i==m) continue;

				dh(m,i) = dhdum(i,0);
				dh(i,m) = dhdum(i,0);
			}*/

			for(int i=0; i<n-1; i++) {
				int ii=i;
				if(i>=m) ii=i+1;
				double temp = lr * ( dhdum(ii,0) - D(ii,m) ) * pow( dhdum(ii,0), 1-r );
				for(int j=0; j<dim; j++) {
					dhmat(i,j) = temp;
				}
			}

			for(int i=0; i<n-1; i++) {
				int ii=i;
				if(i>=m) ii=i+1;
				for(int j=0; j<dim; j++) {
					X(ii,j) += dhmat(i,j) * pow(fabs( pmat(ii,j) ),r-1) * sign(pmat(ii,j));
				}
			}
		}
	}
	return X;
}


int emd_bits(uchar a, uchar b, bool arev=false, bool brev=false) {
	int e = 0; // the resulting distance
	int csa = 0, csb = 0; // cumulative sums
	for (int i=0; i<8; i++) {
		int j = arev ? (1 << (7-i)) : (1 << i);
		int k = brev ? (1 << (7-i)) : (1 << i);
		csa += (a & j) == j;
		csb += (b & k) == k;
		e += abs(csa-csb);
	}
	return e;
}


Mat lbp(const Mat &I) {
    Mat_<uchar> feature(I.size(),0);
    Mat_<uchar> img(I);
    const int m=1;
    for (int r=m; r<img.rows-m; r++) {
        for (int c=m; c<img.cols-m; c++) {
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
            feature(r,c) = v;
        }
    }
    return feature;
}

Mat lut(const Mat &img, const Mat &table) {
	const uchar *input  = img.ptr<uchar>();
	const Vec3b *lookup = table.ptr<Vec3b>();
	Mat res(img.size(),CV_8UC3);
	Vec3b *output = res.ptr<Vec3b>();
	for (size_t i=0; i<img.total(); i++)
		output[i] = lookup[input[i]];
	return res;
}

int test_lbp(const Mat &lookup) {
	Mat face = imread("../face.png", 0);
	Mat fea = lbp(face);
	Mat col = lut(fea, lookup);
	imshow("face",face);
	imshow("fea",fea);
	imshow("col",col);
	waitKey();
	return 0;
}


int main(int argc, const char * argv[]) {
	Mat_<double> regular_dist(256,256,0.0);
	Mat_<double> cyclic_dist(256,256,0.0);
	for (int i=0; i<256; i++) {
		for (int j=0; j<256; j++) {
			if (i==j) continue;
			int a = emd_bits(i,j);
			int b = emd_bits(i,j, true);
			int c = emd_bits(i,j, false, true);
			regular_dist(i,j) = a;
			cyclic_dist(i,j) = min(a,min(b,c));
		}
	}
	Mat X = MDS_UCF(cyclic_dist,3,10);
	// normalize, so we can use it with LUT later:
	normalize(X,X,255,0,NORM_MINMAX, CV_8U);
	cerr << X << endl;
	/*double m,M;
	minMaxLoc(X,&m,&M,0,0);
	X += abs(m);
	X.convertTo(X,CV_8U,255.0/M);
	*/
	test_lbp(X);
    return 0;
}
