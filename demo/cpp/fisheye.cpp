#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;



struct Fish
{
	Mat_<float> map_x, map_y;

	void make_barrel_maps(const Size &s, float kx, float ky)
	{
		map_x.create(s);
		map_y.create(s);

		// the image size
		int w = s.width;
		int h = s.height;
		float Cx = float(w)/2;
		float Cy = float(h)/2;
		for (int y = 0; y < h; y++)
		{
		    int ty= y-Cy;
		    for (int x = 0; x < w; x++)
		    {
		        int tx= x-Cx;
		        int rt= tx*tx+ty*ty;

		        map_x(y,x) = (float)(tx*(1+kx*rt)+Cx);
		        map_y(y,x) = (float)(ty*(1+ky*rt)+Cy);
		    }
		}
	}
	void make_gaussian_maps(const Size &s, double sigma, double mu_x=0, double mu_y=0)
	{
		double sig2 = sigma * sigma;
		double oo_sig2 = sig2 > 0.0000001 ? (1.0/sig2) : 1.0;

		double nrm = sqrt(CV_PI * sig2);
		double oo_nrm = nrm > 0.0000001 ? (1.0/nrm) : 1.0;

		mu_x += s.width/2;
		mu_y += s.height/2;

	    map_x.create(s);
	    map_y.create(s);

		for(int j = 0; j < s.height; j++)
		{
			for(int i = 0; i < s.width; i++)
			{
				double dx = double(i) - mu_x;
				double ex = dx * dx * oo_sig2;
				//double x = exp(-ex) * oo_nrm;
				double x = ex * oo_nrm;
				map_x(j,i) = float(i) + x;

				double dy = double(j) - mu_y;
				double ey = dy * dy * oo_sig2;
				//double y = exp(-ey) * oo_nrm;
				double y = ey * oo_nrm;
				map_y(j,i) = float(j) + y;
			}
		}
	}
	void make_circle_maps(const Size &s)
	{
	    // create destination and the maps
	    //dst.create(src.size(), src.type());
	    map_x.create(s);
	    map_y.create(s);

		double rad = (s.height < s.width ? s.height : s.width)/2;
		double diag_rad = sqrt(s.height*s.height + s.width*s.width)/2;
		printf("radius = %d (rows: %d, cols: %d)\n", (int)rad, s.height, s.width);

		// the center
		double c_x = (double)s.width/2;
		double c_y = (double)s.height/2;

		for(int j = 0; j < s.height; j++) {
			for(int i = 0; i < s.width; i++) {
			// shift the coordinates space to center
			double x = i-c_x;
			double y = j-c_y;

			// handle the 0 and pi/2 angles separately as we are doing atan
			if(0 == x) {
				double ratio = 2*rad/s.height;
				map_y(j,i) = y/ratio + c_y;
				map_x(j,i) = c_x;
			}
			else if(0 == y) {
				double ratio = 2*rad/s.width;
				map_x(j,i) = x/ratio + c_x;
				map_y(j,i) = c_y;
			}
			else {
				// get the r and theta
				double r = sqrt(y*y + x*x);
				double theta = atan(y/x);
				// get the length of line at theta touching the rectangle border
				double diag = min(fabs(c_x/cos(theta)), fabs(c_y/sin(theta)));

				// scale r
				double ratio = rad/diag;
				r = r/ratio;

				// remap the point
				if(x > 0)       map_x(j,i) = r*cos(fabs(theta)) + c_x;
				else            map_x(j,i) = c_x - r*cos(fabs(theta));

				if(y > 0)       map_y(j,i) = r*sin(fabs(theta)) + c_y;
				else            map_y(j,i) = c_y - r*sin(fabs(theta));
				}
			}
		}
	}

	void process(const Mat &f, Mat &f2) {
	    if (f.size() != map_x.size())
//	    	make_gaussian_maps(f.size(), 0.8, 0.0, 0);
//	    	make_circle_maps(f.size());
	    	make_barrel_maps(f.size(), 0.0001f, 0.0001f);
        Mat f1;
        remap(f, f2, map_x, map_y, INTER_NEAREST, BORDER_REFLECT, Scalar(0,0,0));
        /*
        int c=150; // crop
        Rect r(c,c,f.cols-(2*c),f.rows-(2*c));
        resize(f1(r),f2,f.size());
        */
	}
};



int main()
{
	VideoCapture cap(0);
	Fish fish;
	while(cap.isOpened())
	{
		Mat f,f2;
		cap >> f;
		if (f.empty())
		{
			cerr << ".";
			continue;
		}
		fish.process(f,f2);
        imshow("ocv",f2);
		if (waitKey(10)==27) break;
	}
}
