//
// http://www.xujuefei.com/lbcnn.html
//
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
using namespace cv;
using namespace std;

#include "profile.h"

//int NUMPAIRS = 12*12;
int N=32;
float INIT  = 0.05f;
float LEARN = 0.00001f;
float DECAY = 0.96f; // per 1000 samples

Mat normalized(const Mat &in) {
	Scalar m,d;
	meanStdDev(in,m,d);
	Mat out = in - m;
	return out / d[0];
}

struct Digits
{
	const String path_digits = "c:/data/img/digits.png";   //!< this is digits.png from opencv samples
	const int nClass = 10;
    const int SZ = 20;
    Mat digi;

    Digits()
    {
        PROFILE;
        Mat img = imread(path_digits, 0);
        for (int r=0; r<img.rows; r+=SZ)
        {
            for (int c=0; c<img.cols; c+=SZ)
            {
                Mat d = img(Rect(c, r, SZ,SZ));
                d = deskew(d); // in-place.
            }
        }
        img.convertTo(digi, CV_32F, 1.0/255.0);
    }
	Mat deskew(const Mat &img)
	{
	    PROFILEX("deskew");
	    int SZ = img.rows;
	    Moments m = moments(img);
	    if (abs(m.mu02) < 1e-2)
	        return img;
	    double skew = m.mu11 / m.mu02;
	    Mat_<float> M(2,3); M << 1, skew, -0.5*SZ*skew, 0, 1, 0;
	    Mat res;
	    warpAffine(img, res, M, Size(SZ,SZ), WARP_INVERSE_MAP | INTER_LINEAR);
	    return res;
	}
    virtual void train(int n, vector<Mat> &data, Mat &labels) {batch(n,data,labels,0);}
    virtual void test(int n, vector<Mat> &data, Mat &labels) {batch(n,data,labels,50);}
    virtual void batch(int n, vector<Mat> &data, Mat &labels, int off)
    {
        for (int i=0; i<n; i++)
        {
            PROFILEX("digi_img");
            int r = theRNG().uniform(0,50);
            int c = theRNG().uniform(0,50) + off;

            Mat m2, m = digi(Rect(c*SZ,r*SZ,SZ,SZ));
            resize(m,m2,Size(32,32));
            data.push_back(normalized(m2));

            Mat lab(1,10,CV_32F,0.0f);
            lab.at<float>(int(r/5)) = 1.0f;
            labels.push_back(lab);
        }
    }
};

struct AttFaces
{
	const string path_att = "c:/data/faces/att";
    const int nClass = 40;
    const int pSiz = N;
    const int nImg = 10;
    virtual void train(int n, vector<Mat> &data, Mat &labels) {batch(n,data,labels,0);}
    virtual void test(int n, vector<Mat> &data, Mat &labels) {batch(n,data,labels,5);}
    virtual void batch(int n, vector<Mat> &data, Mat &labels, int off)
    {
    	data.clear();
    	labels.release();
        for (int i=0; i<n; i++)
        {
            int p = theRNG().uniform(1, 1+nClass);
            int n = theRNG().uniform(1, 1+(nImg/2)) + off;
            String fn = format("%s/s%d/%d.pgm", path_att.c_str(), p, n);
            Mat m = imread(fn, 0);
            if (m.empty())
            {
                cout << "bad image: " << fn << endl;
                continue;
            }
            Mat um;
            resize(m,m,Size(pSiz,pSiz));
            m.convertTo(um, CV_32F, 1.0/255.0);
            data.push_back(normalized(um));

            Mat lab(1,nClass,CV_32F,0.0f);
            lab.at<float>(p-1) = 1.0f;
            labels.push_back(lab);
        }
    }
};


// activations:
struct linear {
	Mat fw(const Mat &m, float f=1)
	{
	    PROFILE;
	    return (1 + m) * 0.5f;
	}
	Mat bw(const Mat &m)
	{
	    PROFILE;
	    return 1.0f - (m*0.5f);
	}
};

struct absx {
	Mat fw(const Mat &m, float f=1)
	{
	    PROFILE;
	    Mat o = cv::abs(m);
	    return o;
	}
	Mat bw(const Mat &m)
	{
	    return fw(m);
	}
};
struct tan_h {
	Mat fw(const Mat &m, float f=1)
	{
	    PROFILE;
	    Mat _m,ep,en,sp,sn,res;
	    multiply(m, -f, _m);
	    exp(m, ep);
	    exp(_m, en);
	    add(ep, en, sp);
	    subtract(ep, en, sn);
	    divide(sn, sp, res);
	    return res;
	}
	Mat bw(const Mat &m)
	{
	    PROFILE;
	    Mat _m, res;
	    subtract(1, m, _m);
	    sqrt(_m, res);
	    return res;
	}
};
struct relu {
	Mat fw(const Mat &m)
	{
	    PROFILEX("relu.fw");
	    Mat u;
	    max(m, 0, u);
	    return u;
	}
	Mat bw(const Mat &m)
	{
	    PROFILEX("relu.bw");
	    Mat u = Mat::zeros(m.size(), m.type());
	    u.setTo(1, m>0);
	    //u.setTo(0, m<=0);
	    return u;
	}
};
struct sigmoid {
	Mat fw(const Mat &m, float f=1)
	{
	    PROFILE;
	    Mat P;
	    multiply(m, -f, P);
	    exp(P,P);
	    add(P, 1, P);
	    divide(1,P, P);
	    return P;
	}
	Mat bw(const Mat &m)
	{
	    PROFILE;
	    Mat res;
	    subtract(1.0, m, res);
	    res = res.mul(m);
	    return res;
	}
};
struct min_max {
	Mat fw(const Mat &in, float f=1) {
		Mat out;
		normalize(in,out,1,0,NORM_MINMAX);
		return out;
	}
	Mat bw(const Mat &in) {
		return fw(in);
	}
};
struct thresh {
	Mat fw(const Mat &in, float f=1) {
		Mat out(in.size(),in.type(),Scalar(0));
		out.setTo(f, in>0.25);
		return out;
	}
	Mat bw(const Mat &in) {
		return (in);
	}
};

Vec2d softmax(const Mat &A) {
	Point p;
	Vec2d v(0,0);
	minMaxLoc(A,0,&v[1],0,&p);
	v[0] = std::max(p.x,p.y);
	return v;
}

Mat row(const Mat &a) {
	return a.reshape(1,1);
}
Mat col(const Mat &a) {
	return a.reshape(1,a.total());
}

// visualization helpers:
Mat filters(const Mat &W) {
	int B = 2;
	int _w = (int)(sqrt((float)W.cols));
	int _h = (int)(sqrt((float)W.rows+1));
	int S = (_w + 2*B);
	Mat F(S*_h, S*_h, CV_8U, Scalar(128));
	for (int i=0; i<min(W.rows,_h*_h); i++) { // filters
		Mat f,f2 = W.row(i).reshape(1,_w);
		normalize(f2, f, 0, 255, NORM_MINMAX, CV_8U);
		int u = i % _h;
		int v = i / _h;
		f.copyTo(F(Rect(u*S+B, v*S+B, _w, _w)));
	}
	return F;
}
Mat filters(const vector<Mat> &W) {
	int B = 2;
	int _w = W[0].cols;
	int _h = (int)(sqrt((float)(W.size()+1)));
	int S = (_w + 2*B);
	Mat F(S*_h, S*_h, CV_8U, Scalar(128));
	for (int i=0; i<min(int(W.size()),_h*_h); i++) { // filters
		Mat f;
		normalize(W[i], f, 0, 255, NORM_MINMAX, CV_8U);
		int u = i % _h;
		int v = i / _h;
		f.copyTo(F(Rect(u*S+B, v*S+B, _w, _w)));
	}
	return F;
}

void peek(const Mat &A, const String &s) {
	Scalar m,d;
	meanStdDev(A,m,d);
	cerr << s << m[0] << " " << d[0] << " " << norm(A) << " "  << A.reshape(1,1).at<float>(0,4) << " " << A.reshape(1,1).at<float>(0,5) << " " << A.reshape(1,1).at<float>(0,6) << endl;
}

// network layers
struct dense {
	Mat W, cache_u, cache_d;
	tan_h act;

	dense(int h, int w) {
		W = Mat(h, w, CV_32F);
		randu(W, -INIT, INIT);
    	cerr << "W" << W.size() << W.total() << endl;
	}
	Mat forward(const Mat &A) {
		PROFILEX("de.forward")
		// y=W*x
 		Mat o;
		gemm(row(A),W,1.0,noArray(),0,o);
		o = act.fw(o);
		cache_u = A;
		cache_d = o;
		return o;
	}
	Mat backward(const Mat &A) {
    	PROFILEX("de.backward")
		Mat loss = cache_d - A;
		Mat CU = col(cache_u);
		Mat grad = CU * loss;
		W -= LEARN * grad;
		Mat o;
		gemm(A,W,1.0,noArray(),0,o,GEMM_2_T);
		return act.bw(o.reshape(1,N)); // special case !
	}
};
#if 0
struct lbcnn {
	Mat W, grads;
	vector<Mat> cache_u, cache_d;
	vector<Point> pts[2];
	int m,w;
	//absx act;
	//tan_h act;
	relu act;
	//min_max act;
	//linear act;
	void point_pair(Point &a, Point &b) {
	    RNG r(getTickCount());
        do {
	    	a = Point(r.uniform(0,w), r.uniform(0,w));
		    b = Point(r.uniform(0,w), r.uniform(0,w));
		} while (a.x==b.x && a.y==b.y);
	}
	lbcnn(const Size &s) {
	    init(s,8,3);
	    pts[0] = vector<Point>{Point(0,0), Point(1,0), Point(2,0), Point(2,1), Point(2,2), Point(1,2), Point(0,2), Point(0,1)};
	    pts[1] = vector<Point>{Point(1,1), Point(1,1), Point(1,1), Point(1,1), Point(1,1), Point(1,1), Point(1,1), Point(1,1)};
	}
	lbcnn(const Size &s, int m, int w) {
	    init(s,m,w);
	    pts[0].resize(m);
	    pts[1].resize(m);
	    for (int i=0; i<m; i++) {
	    	point_pair(pts[0][i], pts[1][i]);
		}
	}
	void init(const Size &s, int m, int w) {
	    RNG r(theRNG());//(getTickCount());
		this->w = w;
		this->m = m;
	    W = Mat(m, s.area(), CV_32F, 0.0f); // bernoulli
    	W.forEach<float>([&](float &p, const int*){
    		p = (r.uniform(0,2)>0 ? INIT : -INIT);
    	});
    	cerr << "W" << W.size() << W.total() << endl;
		cache_u.resize(m);
		cache_d.resize(m+1); // +1 for final output
		grads = Mat(W.size(), W.type());
	}
	void reset(int i) {
	    RNG r(getTickCount());
    	pts[0][i] = Point(r.uniform(0,w), r.uniform(0,w));
	    pts[1][i] = Point(r.uniform(0,w), r.uniform(0,w));
    	W.row(i).forEach<float>([&](float &p, const int*){
    		p = (r.uniform(0,2)>0 ? INIT : -INIT);
    	});
    	cerr << "reset " << i << endl;
	}
	Mat conv(const Mat &I, Point p, Point q) const {
		PROFILEX("lb.conv");
	    int M = I.rows;
	    int N = I.cols;
	    Mat ma = I(Range(p.y, p.y+M-w), Range(p.x, p.x+N-w));
	    Mat mb = I(Range(q.y, q.y+M-w), Range(q.x, q.x+N-w));
		Mat sub(I.size(), I.type()); // I.size == O.size
		Mat subroi(sub, Rect(w/2, w/2, N-w, M-w));
		{
			PROFILEX("lb.conv.sub");
			subtract(ma, mb, subroi);
		}
		return sub;
	}
	Mat forward(const Mat &A) {
	    PROFILEX("lb.forward");
	   	cache_u.resize(m);
	    Mat S(A.size(), CV_32F, 0.0f);
	    for (int i=0; i<W.rows; i++) {
	    	Mat c = conv(A, pts[0][i], pts[1][i]);
			cache_u[i] = c.clone();
		    c = act.fw(c);
	 	    {
			    PROFILEX("lb.fw.mul")
			    multiply(c, W.row(i).reshape(1,32), c);
	 	    }
	 		cache_d[i] = c;
		    {
			    PROFILEX("lb.fw.add")
				//scaleAdd(c, 1.0/W.rows, S, S);
			    add(S, c, S);
			}
		}
		//S = act.fw(S);
		normalize(S, S, 1, 0, NORM_MINMAX);
		cache_d[m] = S;
		return S;
	}
	Mat backward(const Mat &A) {
		PROFILEX("lb.backward")
		Mat residual, dx, b(A.size(), CV_32F, 0.0f);
		for (int i=0; i<W.rows; i++) {  // vanilla SGD
			{
				PROFILEX("lb.bw.resi")
				subtract(cache_d[i], A, residual);
			}
			{
				PROFILEX("lb.bw.gemm")
				dx = grads.row(i).reshape(1,32);
				gemm(cache_u[i], residual, 1, noArray(), 0, dx);
			}
			/*{
				PROFILEX("lb.bw.normalize")
				normalize(dx, dx, 0, 1, NORM_MINMAX);
			}*/
			{
				PROFILEX("lb.bw.update")
				scaleAdd(row(dx), -LEARN, W.row(i), W.row(i));

				/*int b=8;
				Mat q = W.row(i).reshape(1,32)(Rect(b,b,32-b,32-b));
				if (abs(mean(q)[0]) <= DBL_MIN) {
					reset(i);
				}*/
			}
		}
		return act.bw(b);
	}
};
#endif
struct rbm {
	Mat weights, grad, hidden;
	vector<Mat> cache_u, cache_d, cache_h;
	//tan_h act;
	//absx act;
	//	sigmoid act;
	//relu act;
	min_max act;
	//thresh act;
	//linear act;
    rbm(Size siz) {
    	RNG r=RNG(17);
    	int nvisible = siz.height * siz.width;
    	int nhidden = nvisible;
        weights = Mat(nvisible, nhidden, CV_32F); // bernoulli
    	weights.forEach<float>([&](float &p, const int*){
    		p = (r.uniform(0,2)>0 ? INIT : -INIT);
    	});
    	cerr << "W" << weights.size() << weights.total() << endl;
		cache_u.resize(1);
		cache_d.resize(1);
		cache_h.resize(1);
		grad = Mat(weights.size(), weights.type(), 0.0f);
	}

    Mat pass(const Mat &m, const Mat &w, int flag=0)
    {
        PROFILEX("rbm.pass")
        Mat res;
        //cerr << "p " << m.size() << " x " << w.size() << " " << flag << endl;
        gemm(m, w, 1, noArray(), 0, res, flag);
        return  res; //act.fw(res);
    }

    Mat dream_wake(const Mat &m)
    {
        PROFILEX("rbm.dream_wake")
        hidden = act.fw(pass(row(m),weights)) ;

        return act.bw(pass(hidden, weights, GEMM_2_T));
    }

	Mat forward(const Mat &A) {
	    PROFILEX("rbm.forward");
		cache_h[0] = hidden.clone();
		cache_u[0] = A.reshape(1,1);
		cache_d[0] = dream_wake(A);
		//cerr << "fw ok. " << A.size() << " " << dn.size() << " " << hidden.size() << endl;
		return cache_d[0];
	}
	Mat backward(const Mat &A) {
		PROFILEX("rbm.backward")
        Mat residual = cache_d[0] - A.reshape(1,1);// = pass(A,cache_h[0]);
		//cerr << "bw " << A.size() << " " << cache_d[0].size() <<  " " << cache_u[0].size() << " " << residual.size() << endl;
        //Mat dx1 = pass(row(A), up, GEMM_1_T);
       // Mat dx2 = pass(col(cache_d[0]), up, 0);
        //gemm(row(A), up, 1, noArray(), 0, dx2, GEMM_1_T);
        Mat dx1 = pass(col(cache_u[0]), residual, 0);
        grad = dx1;// - dx2;
        //grad = act.bw(dx1);
        //normalize(grad,grad);
		//cerr << "grad " << grad.size() <<  " " << weights.size() << endl;
        //normalize(dx1,dx1);
        //normalize(dx2,dx2);
        //subtract(dx1, dx2, grad);

		scaleAdd(grad, -LEARN, weights, weights);
		//cerr << "bw ok. " << grad.size()  << endl;

		return dream_wake(A);
	}
};

int main(int argc, char **argv) {
	bool useOCL = argc > 1;
    ocl::setUseOpenCL(useOCL);
    cout << "ocl " << cv::ocl::useOpenCL() << endl;

    namedWindow("act", 0);
    namedWindow("grads", 0);
    namedWindow("dense", 0);
    namedWindow("weights", 0);
    namedWindow("hidden", 0);
    //namedWindow("upstream", 0);
    //namedWindow("downstream", 0);

    AttFaces problem;
	//Digits problem;

    vector<Mat> train_data;
    Mat train_labels;
    problem.train(400, train_data, train_labels);
    vector<Mat> test_data;
    Mat test_labels;
    problem.test(400, test_data, test_labels);

    Size siz(N,N);
    rbm l1(siz);
    rbm l2(siz);
    rbm l3(siz);
    dense d1(N*N, problem.nClass);

    int test = 0;
    int correct = 0;
    int maxgen = 500000;
	for (int i=1; i<maxgen; i++) {
		PROFILEX("gen")
		Mat A = train_data[i%400];
	    Mat T = train_labels.row(i%400);

	    Mat B = l1.forward(A);
	    Mat C = l2.forward(B);
	    Mat D = l3.forward(C);
	    Mat P = d1.forward(D);

	    Vec2d p = softmax(P);
	    Vec2d t = softmax(T);
	    correct += p[0]==t[0];

	    Mat Dr = d1.backward(T);
	    Mat Cr = l3.backward(Dr);
	    Mat Br = l2.backward(Cr);
	    Mat Ar = l1.backward(Br);

	    if (i%47==0)
	    {
	    	PROFILEX("draw")
			cerr << i << " " << p << t << " " << correct << " " << (float(correct)/i) << '\n';
			Mat act=A.clone();
			putText(act,format("%d", int(t[0])),Point(10,10),1,1,Scalar(0));
			hconcat(act,B.reshape(1,N),act);
			hconcat(act,C.reshape(1,N),act);
			hconcat(act,D.reshape(1,N),act);

			Mat r=Ar.clone().reshape(1,N);
			hconcat(r,Br.reshape(1,N),r);
			hconcat(r,Cr.reshape(1,N),r);
			hconcat(r,Dr.reshape(1,N),r);

			vconcat(act,r,act);
			resize(act,act,Size(), 2,2);
			imshow("act",act);

			Mat f = filters(l1.weights);
			hconcat(f,filters(l2.weights),f);
			hconcat(f,filters(l3.weights),f);
			resize(f,f,Size(), 2,2);
			imshow("weights",f);

			Mat g = filters(l1.grad);
			hconcat(g,filters(l2.grad),g);
			hconcat(g,filters(l3.grad),g);
			resize(g,g,Size(), 2,2);
			imshow("grads",g);

			Mat h=l1.hidden.clone().reshape(1,N);
			hconcat(h,l2.hidden.reshape(1,N),h);
			hconcat(h,l3.hidden.reshape(1,N),h);
			imshow("hidden",h);

			imshow("dense",filters(d1.W.t()));

			int k = waitKey(5);
			if (k==27) return 0;
			if (k==' ') waitKey(5000);
			if (k=='t') { test = 1; }
		}
		if (test || (i%1000==0)) {
			int correct=0;
			for (int j=0; j<test_data.size(); j++) {
			    Mat B = l1.forward(test_data[j]);
			    Mat C = l2.forward(B);
			    Mat D = l3.forward(C);
				Mat P = d1.forward(D);
			    Mat T = test_labels.row(j);
			    Vec2d p = softmax(P);
			    Vec2d t = softmax(T);
			    correct += p[0]==t[0];
			}
			if (LEARN>0.000005)
				LEARN *= DECAY;
			cerr << "TEST " << correct << " " << (float(correct)/test_data.size()) << " " << LEARN << endl;
			test = 0;
		}
	}

	waitKey();
	return 0;
}
