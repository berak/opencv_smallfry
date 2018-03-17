#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/optflow.hpp"

using namespace std;
using namespace cv;

int writepipe(const Mat &m) {
	const Vec3b *pixel=m.ptr<Vec3b>();
	for (size_t i=0; i<m.total(); i++) {
		const Vec3b &p = pixel[i];
		putc(p[0],stdout);
		putc(p[1],stdout);
		putc(p[2],stdout);
	}
}

Mat readpipe(int w, int h) {
    Mat m(h,w,CV_8UC3);
	Vec3b *pixel=m.ptr<Vec3b>();
	for (size_t i=0; i<m.total(); i++) {
		Vec3b &p = pixel[i];
		p[0] = getc(stdin);
		p[1] = getc(stdin);
		p[2] = getc(stdin);
	}
    return m;
}
int main(int argc, char**argv)
{
    Mat m(1080,1920,CV_8UC3);
    randu(m,Scalar::all(0),Scalar::all(255));
    int64 t0 = getTickCount();
    for (int n=0; n<10; n++) {
    	if (argc>1)
    		readpipe(m.cols, m.rows);
    	else
    		writepipe(m);
    }
    int64 t1 = getTickCount();
    cerr << argc << " took " << (t1-t0)/getTickFrequency() << " seconds." << endl;
    return 0;
}
