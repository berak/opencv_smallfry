//
// from: https://github.com/LaurentBerger/FourierDescriptors/
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

namespace fourier {

class MatchDescriptor {
public :
    vector<Point2d> sContour;
    vector<complex<float> > a, b;
    static vector<float> frequence;
    vector<float> rho,psi;
    double pi;
    int nbDesFit;

public :
    MatchDescriptor(){nbDesFit=7;pi=acos(-1.0);};;
    float AjustementRtSafe(const vector<Point2d> &c,float &alphaMin,float &phiMin,float &sMin);
    void falpha(float x,float *fn,float *df);
    void InitFrequence();
    float rtsafe(float x1,float x2,float xacc);
    float Distance(complex<float> r,float alpha)
    {
	    long			i;
	    complex<float>	j(0,1);
	    float 			d=0;

	    for (i=1; i<=nbDesFit; i++)
	    {
	        d += abs(a[i]-b[i]*r*exp(j*float(alpha*frequence[i])))+ abs(a[a.size()-i]-b[a.size()-i]*r*exp(j*float(alpha*frequence[a.size()-i])));
	    }
	    return d;
    }
};
vector<float> MatchDescriptor::frequence;

void MatchDescriptor::InitFrequence()
{
	long i;
	int nbElt=sContour.size();
	frequence.resize(sContour.size());

	for (i=0;i<=nbElt/2;i++)
		frequence[i] = 2*pi*(float)i/nbElt;
	for (i=nbElt/2+1;i<nbElt;i++)
		frequence[i] = 2*pi*(float)(i-nbElt)/nbElt;
}


void MatchDescriptor::falpha(float x,float *fn,float *df)
{
	long 	n,nbElt = sContour.size();
	float	s1=0,s2=0,s3=0,s4=0;
	float	ds1=0,ds2=0,ds3=0,ds4=0;

	for (n=1; n<=nbDesFit; n++)
	{
		s1 += 	rho[n] * sin(psi[n]+frequence[n]*x) +
				rho[nbElt-n] * sin(psi[nbElt-n]+frequence[nbElt-n]*x);
		s2 += 	frequence[n] * rho[n] * cos(psi[n]+frequence[n]*x) +
				frequence[nbElt-n] * rho[nbElt-n] * cos(psi[nbElt-n]+frequence[nbElt-n]*x);
		s3 += 	rho[n] * cos(psi[n]+frequence[n]*x) +
				rho[nbElt-n] * cos(psi[nbElt-n]+frequence[nbElt-n]*x);
		s4 += 	frequence[n] * rho[n] * sin(psi[n]+frequence[n]*x) +
				frequence[nbElt-n] * rho[nbElt-n] * sin(psi[nbElt-n]+frequence[nbElt-n]*x);
		ds1 += 	frequence[n]*rho[n] * cos(psi[n]+frequence[n]*x) +
		 		frequence[nbElt-n]*rho[nbElt-n] * cos(psi[nbElt-n]+frequence[nbElt-n]*x);
		ds2 += 	-frequence[n]*frequence[n] * rho[n] * sin(psi[n]+frequence[n]*x) -
				frequence[nbElt-n]*frequence[nbElt-n] * rho[nbElt-n] * sin(psi[nbElt-n]+frequence[nbElt-n]*x);
		ds3 += 	-frequence[n]*rho[n] * sin(psi[n]+frequence[n]*x) -
				frequence[nbElt-n]*rho[nbElt-n] * sin(psi[nbElt-n]+frequence[nbElt-n]*x);
		ds4 += 	frequence[n]*frequence[n] * rho[n] * cos(psi[n]+frequence[n]*x) +
		 		frequence[nbElt-n]*frequence[nbElt-n] * rho[nbElt-n] * cos(psi[nbElt-n]+frequence[nbElt-n]*x);
	}
	*fn = s1 * s2 - s3 *s4;
	*df = ds1 * s2 + s1 * ds2 - ds3 * s4 -  s3 * ds4;
}

float MatchDescriptor::AjustementRtSafe(const vector<Point2d> &c,float &alphaMin,float &phiMin,float &sMin)
{
	long		    n,nbElt = sContour.size();
	float 		    s1,s2,sign1,sign2,df,x1=nbElt,x2=nbElt,dx;
	float		    dist,distMin = 10000,alpha,s,phi;
	complex<float> 	j(0,1),zz;

	InitFrequence();
	rho.resize(nbElt);
	psi.resize(nbElt);

	b.resize(nbElt);
	a.resize(nbElt);
	if (nbElt != c.size())
		return -1;
	for (n=0; n<nbElt; n++)
	{
		b[n] = complex<float>(sContour[n].x, sContour[n].y);
	    a[n] = complex<float>(c[n].x, c[n].y);
		zz = conj(a[n]) * b[n];
		rho[n] = abs(zz);
		psi[n] = arg(zz);
	}
	float xp=-nbElt,fnp,dfp;
	falpha(xp,&fnp,&dfp);

	x1=nbElt,x2=nbElt;
	sMin =1;
	alphaMin = 0;
	phiMin = arg(a[1]/b[1]);
	do
	{
		x2 = x1;
	    falpha(x2,&sign2,&df);
		dx = 1;
		x1 = x2;
		do
		{
	        x2=x1;
			x1 -= dx;
	        falpha(x1,&sign1,&df);
		}
		while ((sign1*sign2>0)&&(x1>-nbElt));
		if (sign1*sign2<0)
		{
			alpha = rtsafe(x1, x2, 1e-8);
	        falpha(alpha, &sign1, &df);
	        alpha = alpha;
			s1 = 0;
			s2 = 0;
			for (n=1; n<nbElt; n++)
			{
				s1 += rho[n] * sin(psi[n] + frequence[n] * alpha);
				s2 += rho[n] * cos(psi[n] + frequence[n] * alpha);
			}
		    phi = -atan2(s1,s2);
			s1 = 0;
			s2 = 0;
	        for (n=1; n<nbElt; n++)
	        {
				s1 += rho[n] * cos(psi[n] + frequence[n] * alpha + phi) ;
				s2 += abs(b[n] * conj(b[n]));
	        }
			s = s1/s2;
			zz = s*exp(j*phi);
			if (s>0)
				dist = Distance(zz, alpha);
			else
				dist = 10000;
			if (dist < distMin)
			{
				distMin = dist;
				alphaMin = alpha;
				phiMin = phi;
				sMin = s;
			}
		}
	}
	while ((x1 > -nbElt));
	return distMin;
}


#define MAXIT 100

float MatchDescriptor::rtsafe(float x1,float x2,float xacc)
{
	long j;
	float df,dx,dxold,f,fh,fl;
	float temp,xh,xl,rts;

	falpha(x1,&fl,&df);
	falpha(x2,&fh,&df);
	if (fl < 0.0)
	{
		xl=x2;
		xh=x1;
	}
	else
	{
		xh=x2;
		xl=x1;
	}
	rts=0.5*(x1+x2);
	dxold=fabs(x2-x1);
	dx=dxold;
	falpha(rts,&f,&df);
	for (j=1;j<=MAXIT;j++)
	{
		if ((((rts-xh)*df-f)*((rts-xl)*df-f) >= 0.0)
			|| (fabs(2.0*f) > fabs(dxold*df)))
		{
			dxold=dx;
			dx=0.5*(xh-xl);
			rts=xl+dx;
			if (xl == rts) return rts;
		}
		else
		{
			dxold=dx;
			dx=f/df;
			temp=rts;
			rts -= dx;
			if (temp == rts) return rts;
		}
		if (fabs(dx) < xacc) return rts;

		falpha(rts,&f,&df);
		if (f < 0.0)
			xl=rts;
		else
			xh=rts;
	}
	return 0.0;
}


Point2d Echantillon(const vector<Point> &c,long i,float l1,float l2,float s)
{
	Point2d d = c[(i+1) % c.size()] - c[i % c.size()];
	Point2d p = Point2d(c[i % c.size()]) + d * (s-l1)/(l2-l1);
	return p;
}


vector<Point2d> ReSampleContour(const vector<Point> &c, int nbElt)
{
	long 		nb=c.size();
	float		l1=0,l2,p,d,s;
	vector<Point2d> r;
	int j=0;
	p = arcLength(c, true);

	l2 = norm(c[j] - c[j+1]) / p;
	for(int i=0; i<nbElt; i++)
	{
		s = (float)i/(float)nbElt;
		while (s >= l2)
		{
			j++;
			l1 = l2;
			d = norm(c[j % nb] - c[(j+1) % nb]);
			l2 = l1+d/p;
		}
		if ((s >= l1) && (s < l2))
			r.push_back(Echantillon(c, j, l1, l2, s));
	}
	return r;
}


vector<Point2d> zsample(const vector<Point> &a, int n=1024)
{
	vector<Point2d> Z,z = ReSampleContour(a, n);
    dft(z, Z, DFT_SCALE|DFT_REAL_OUTPUT);
    return Z;
}

double distance(const vector<Point2d> &a,const vector<Point2d> &b, float &alpha, float &phi, float &s)
{
	MatchDescriptor md;
	md.sContour = a;
	md.nbDesFit = 20;

    md.AjustementRtSafe(b, alpha, phi, s);
    complex<float> expitheta = s * complex<float>(cos(phi), sin(phi));

    return md.Distance(expitheta, alpha);
}

double distance(const vector<Point> &a,const vector<Point> &b, float &alpha, float &phi, float &s)
{   const int n = 1024;
	vector<Point2d> Za = zsample(a, n);
	vector<Point2d> Zb = zsample(b, n);
    return distance(Za,Zb, alpha,phi,s);
}

double distance(const vector<Point> &a,const vector<Point> &b)
{
	float alpha, phi, s;
	return distance(a, b, alpha, phi, s);
}

// from Z
// phi is the angle from a to b, so if you want to reconstruct b from a, reverse phi.
vector<Point> reconstruct(const vector<Point2d> &Za, float alpha, float phi, float s)
{
    complex<float> expitheta = s * complex<float>(cos(phi), sin(phi));
    vector<Point2d> Zb(Za.size());
    for (int j=1; j<Zb.size(); j++)
    {
        complex<float> zr(Za[j].x, Za[j].y);
        zr = zr * expitheta * exp(alpha * MatchDescriptor::frequence[j] * complex<float>(0,1));
        Zb[j].x = zr.real();
        Zb[j].y = zr.imag();
    }
    vector<Point2d> z;
    dft(Zb, z, DFT_INVERSE);
    vector<Point> c;
    for (int j = 0; j<z.size(); j++)
        c.push_back(z[j]);
    return (c);
}

// from contour
vector<Point> reconstruct(const vector<Point> &a, float alpha, float phi, float s)
{
	return reconstruct(zsample(a),alpha,phi,s);
}

}; // namespace fourier



#include "matching.h"
namespace fourier {
	using namespace matching;

	struct MatcherImpl : Matcher {
		int N; // resample contours to N fixed size
		std::vector<vector<Point2d>> shapes;
		MatcherImpl(int n) : N(n) {}

		virtual void add(const vector<Point> &p) {
			vector<Point2d> z = fourier::zsample(p, N);
			shapes.push_back(z);
		}

		virtual void match(const vector<Point> &p, vector<Point2d> &best, double &dist, int &id){
			vector<Point2d> Z = fourier::zsample(p, N); // compare in Z space
			dist=99999999;
			id=-1;
			for (size_t i=0; i<shapes.size(); i++) {
				float a,p,s;
				double d = fourier::distance(shapes[i], Z, a,p,s);
				if (d < dist) {
					dist = d;
					id = i;
				}
			}
			if (id!=-1) {
				best = shapes[id];
			}
		}
	};
	cv::Ptr<Matcher> createMatcher(int n) {
		return cv::makePtr<MatcherImpl>(n);
	}
}

/*
int main(int argc, char **argv)
{
	Mat m1,m2,mc;
	Mat m = imread("scene1.jpg", IMREAD_GRAYSCALE);
	cvtColor(m/8, mc, COLOR_GRAY2BGR); // for drawing
	bitwise_not(m, m1);
	blur(m1,m1,Size(6,6));
	threshold(m1, m2, 5, 255, THRESH_BINARY);

	vector<vector<Point> > raw_contours, contours;
	findContours(m2, raw_contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	// filter for size
	for (int i=0; i<raw_contours.size(); i++)
	{
	    if (raw_contours[i].size() >= 200 )
	    {
	        contours.push_back(raw_contours[i]);
	        drawContours(mc, raw_contours, i, Scalar(128,128,128), 2);
	    }
	}
	cerr << contours.size() << " filtered contours." << endl;

	int  ca=3, cb=2;
	float alpha, phi, s; // needed for reconstruction later
	double d = fourier::distance(contours[ca], contours[cb], alpha, phi, s);
    cout <<"Self Distance for contour " << ca << " and " << ca << " is " << fourier::distance(contours[ca], contours[ca]) << endl;
    cout <<"Self Distance for contour " << cb << " and " << cb << " is " << fourier::distance(contours[cb], contours[cb]) << endl;
    drawContours(mc, contours, ca, Scalar(0,0,255), 2);
    drawContours(mc, contours, cb, Scalar(0,255,0), 2);
    cout <<"Distance between contour " << ca << " and " << cb << " is " << d << " " << endl;

	//vector<Point> rec = fourier::reconstruct(contours[cb],alpha,-phi,s); // a from b
	vector<Point> rec = fourier::reconstruct(contours[ca], alpha, phi, s); // b from a
    int last = contours.size();
    contours.push_back(rec);
    drawContours(mc, contours, last, Scalar(255,0,0), 2, LINE_AA, noArray(), INT_MAX, Point(120,120));
	imshow("mc ",mc);
	waitKey();

	return 0;
};

*/
