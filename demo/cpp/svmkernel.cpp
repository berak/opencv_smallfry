#include <opencv2/opencv.hpp>
#include <opencv2/cvconfig.h>
using namespace std;
using namespace cv;


Mat normalized(const Mat &in) {
    Scalar m,d;
    meanStdDev(in,m,d);
    Mat out = in - m;
    return out / d[0];
}

struct Digits
{
    const String path_digits = "c:/data/img/digits.png";   //!< this is digits.png from opencv samples
    const int nClass = 10; // 5 rows, 100 cols per class
    const int SZ = 20;
    Mat digi;

    Digits()
    {
        Mat img = imread(path_digits, 0);
        img.convertTo(digi, CV_32F, 1.0/255.0);
    }
    virtual void train(int n, Mat &data, Mat &labels) {batch(n,data,labels,0);}
    virtual void test(int n, Mat &data, Mat &labels) {batch(n,data,labels,50);}
    virtual void batch(int n, Mat &data, Mat &labels, int off)
    {
        for (int i=0; i<n; i++)
        {
            int r = theRNG().uniform(0,50);
            int c = theRNG().uniform(0,50) + off;

            Mat m2, m = digi(Rect(c*SZ,r*SZ,SZ,SZ));
            resize(m,m2,Size(32,32));
            data.push_back(normalized(m2).reshape(1,1));
            labels.push_back(int(r/5));
        }
    }
};

struct XKernel : public ml::SVM::Kernel
{
    virtual ~XKernel() {}

    //! your distance metric between one of the support vecs(sample) and your query(another) goes here
    virtual float per_elem(int var_count, const float* sample, const float* another) = 0;
    //! post-process results array (if nessecary)
    virtual void post(int vcount, float* results) {}

    void calc(int vcount, int var_count, const float* vecs, const float* another, float* results)
    {
        for (int j=0; j<vcount; j++)
        {
            const float* sample = &vecs[j*var_count];
            results[j] = per_elem(var_count, sample, another);
        }
        post(vcount, results);
    }

    int getType(void) const
    {
        return -1; // we're special.
    }

    float sse_ret(__m128 s)
    {
        union { __m128 m; float f[4]; } x;
        x.m = s;
        return (x.f[0] + x.f[1] + x.f[2] + x.f[3]);
    }

    float dot(int var_count, const float* sample, const float* another)
    {
        __m128  d, s = _mm_set_ps1(0);
        __m128* anot = (__m128*)another;
        __m128* samp = (__m128*)sample;
        for (int k=0; k<var_count; k+=4, anot++, samp++)
        {
            d = _mm_mul_ps(*anot, *samp);
            s = _mm_add_ps(s, d);
        }
        return sse_ret(s);
    }

    float l2sqr(int var_count, const float* sample, const float* another)
    {
        __m128  d, s = _mm_set_ps1(0);
        __m128* anot = (__m128*)another;
        __m128* samp = (__m128*)sample;
        for (int k=0; k<var_count; k+=4, anot++, samp++)
        {
            d = _mm_sub_ps(*samp, *anot);
            d = _mm_mul_ps(d, d);
            s = _mm_add_ps(s, d);
        }
        return sse_ret(s);
    }
};


struct linear_sse : public XKernel
{
    float per_elem(int var_count, const float* sample, const float* another)
    {
        return XKernel::dot(var_count, sample, another);
    }
    static Ptr<ml::SVM::Kernel> create() { return makePtr<linear_sse>(); }
};

struct inter_sse : public XKernel
{
    float per_elem(int var_count, const float *sample, const float *another)
    {
        __m128  c, s = _mm_set_ps1(0);
        __m128* anot = (__m128*)another;
        __m128* samp = (__m128*)sample;
        for (int k=0; k<var_count; k+=4, anot++, samp++)
        {
            c = _mm_min_ps(*anot, *samp);
            s = _mm_add_ps(s, c);
        }
        return sse_ret(s);
    }
    static Ptr<ml::SVM::Kernel> create() { return makePtr<inter_sse>(); }
};


struct chi2_sse : public XKernel
{
    double gamma;
    chi2_sse(double gamma) : gamma(gamma) {}
    float per_elem(int var_count, const float* sample, const float* another)
    {
        __m128 d, D, e, s = _mm_set_ps1(0);
        __m128* anot = (__m128*)another;
        __m128* samp = (__m128*)sample;
        for (int k=0; k<var_count; k+=4, anot++, samp++)
        {
            d  = _mm_sub_ps(*anot, *samp);
            D  = _mm_add_ps(*anot, *samp);
            e  = _mm_mul_ps(d, d);
            s += _mm_div_ps(e, D);
        }
        return -gamma * sse_ret(s);
    }
    void post(int vcount, float* results)
    {
        if (vcount <= 0) return;
        Mat R( 1, vcount, CV_32F, results );
        exp( R, R );
        min( R, (FLT_MAX*1e-3), R ); // cleanup behind exp()
    }
    static Ptr<ml::SVM::Kernel> create(double gamma) { return makePtr<chi2_sse>(gamma); }
};


struct poly_sse : public XKernel
{
    double gamma, degree, coef0;
    poly_sse(double gamma, double degree, double coef0=0) : gamma(gamma), degree(degree), coef0(coef0) {}
    float per_elem(int var_count, const float* sample, const float* another)
    {
        return gamma * XKernel::dot(var_count, sample, another) + coef0;
    }
    void post(int vcount, float* results)
    {
        if (vcount <= 0) return;
        Mat R( 1, vcount, CV_32F, results );
        pow( R, degree, R );
        min( R, (FLT_MAX*1e-3), R ); // cleanup behind pow()
    }
    static Ptr<ml::SVM::Kernel> create(double gamma, double degree, double coef0=0) { return makePtr<poly_sse>(gamma, degree, coef0); }
};

struct sigmoid_sse : public XKernel
{
    double gamma, coef0;
    sigmoid_sse(double gamma, double coef0=0) : gamma(gamma), coef0(coef0) {}
    float per_elem(int var_count, const float* sample, const float* another)
    {
        return -2*gamma * XKernel::dot(var_count, sample, another) - 2*coef0;
    }
    void post(int vcount, float* results)
    {
        for( int j = 0; j < vcount; j++ )
        {
            float t = results[j];
            float e = std::exp(-std::abs(t));
            if( t > 0 )
                results[j] = (float)((1. - e)/(1. + e));
            else
                results[j] = (float)((e - 1.)/(e + 1.));
        }
    }
    static Ptr<ml::SVM::Kernel> create(double gamma, double coef0) { return makePtr<sigmoid_sse>(gamma, coef0); }
};


struct rbf_sse : public XKernel
{
    double gamma;
    rbf_sse(double gamma) : gamma(gamma) {}
    float per_elem(int var_count, const float* sample, const float* another)
    {
        return -gamma * XKernel::l2sqr(var_count, sample, another);
    }
    void post(int vcount, float* results)
    {
        if (vcount <= 0) return;
        Mat R( 1, vcount, CV_32F, results );
        exp( R, R );
        min( R, (FLT_MAX*1e-3), R ); // cleanup behind exp()
    }
    static Ptr<ml::SVM::Kernel> create(double gamma) { return makePtr<chi2_sse>(gamma); }


    void calc_rbf( int vcount, int var_count, const float* vecs,
                   const float* another, float* results )
    {
        int j, k;

        for( j = 0; j < vcount; j++ )
        {
            const float* sample = &vecs[j*var_count];
            double s = 0;

            for( k = 0; k <= var_count - 4; k += 4 )
            {
                double t0 = sample[k] - another[k];
                double t1 = sample[k+1] - another[k+1];

                s += t0*t0 + t1*t1;

                t0 = sample[k+2] - another[k+2];
                t1 = sample[k+3] - another[k+3];

                s += t0*t0 + t1*t1;
            }

            for( ; k < var_count; k++ )
            {
                double t0 = sample[k] - another[k];
                s += t0*t0;
            }
            results[j] = (float)(s * -gamma);
        }

        if( vcount > 0 )
        {
            Mat R( 1, vcount, CV_32F, results );
            exp( R, R );
        }
    }
    static void test() {
        rbf_sse r(0.007);
        int K = 500;
        int M = 1024;
        Mat_<float> sv(K,M), an(1,M),rs(1,K),rs2(1,K);
        randu(sv,-1,1);
        randu(an,-1,1);
        r.calc_rbf(K, M, sv.ptr<float>(), an.ptr<float>(), rs.ptr<float>());
        //cout << rs << endl;
        r.calc(K, M, sv.ptr<float>(), an.ptr<float>(), rs2.ptr<float>());
        //cout << rs2 << endl;
        //cout << (rs2-rs) << endl;
        cout << sum(rs2-rs)[0] << endl;
    }
};



int main(int argc, char** argv)
{
    //rbf_sse::test();
    //return 0;
    CommandLineParser parser(argc,argv,
        "{h help ? || print help msg}"
        "{n |500| number of train/test samples}"
        "{t |1| number of threads}"
        "{d |2.3| degree}"
        "{g |0.5| gamma}"
        "{C |1.0| C}"
        "{c coef0 |0.0| coef0}"
        "{k kernel |linear| svm kernels, one of linear,linear_sse,rbf,rbf_sse,sigmoid,sigmoid_sse,poly,poly_sse,inter,inter_sse,chi2,chi2_sse}"
    );
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    int N = parser.get<int>("n");
    int T = parser.get<int>("t");
    double C = parser.get<double>("C");
    double coef0 = parser.get<double>("coef0");
    double degree = parser.get<double>("d");
    double gamma = parser.get<double>("g");
    setNumThreads(T);
    String meth = parser.get<String>("kernel");
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setKernel(ml::SVM::LINEAR);
    svm->setDegree(degree);
    svm->setGamma(gamma);
    svm->setC(C);

    if (meth=="rbf")     { svm->setKernel(ml::SVM::RBF); }
    if (meth=="poly")    { svm->setKernel(ml::SVM::POLY); }
    if (meth=="linear")  { svm->setKernel(ml::SVM::LINEAR); }
    if (meth=="sigmoid") { svm->setKernel(ml::SVM::SIGMOID); }
    if (meth=="chi2")    { svm->setKernel(ml::SVM::CHI2); } // it sucks.
    if (meth=="inter")   { svm->setKernel(ml::SVM::INTER); }
    if (meth=="linear_sse") { svm->setCustomKernel(linear_sse::create()); }
    if (meth=="inter_sse")  { svm->setCustomKernel(inter_sse::create()); }
    if (meth=="chi2_sse")   { svm->setCustomKernel(chi2_sse::create(gamma)); }
    if (meth=="poly_sse")   { svm->setCustomKernel(poly_sse::create(gamma, degree, coef0)); }
    if (meth=="rbf_sse")    { svm->setCustomKernel(rbf_sse::create(gamma)); }
    if (meth=="sigmoid_sse")    { svm->setCustomKernel(sigmoid_sse::create(gamma,coef0)); }

    Digits problem;

    Mat train_data, train_labels;
    problem.train(N, train_data, train_labels);

    Mat test_data, test_labels, test_pred;
    problem.test(N, test_data, test_labels);
    int64 t0 = getTickCount();

    bool ok = svm->train(train_data, ml::ROW_SAMPLE, train_labels);
    if (!ok) { cerr << endl << meth << " train failed" << endl; return -1; }
    int64 t1 = getTickCount();
    double tt = ((t1-t0)/getTickFrequency());

    svm->predict(test_data, test_pred);
    int64 t2 = getTickCount();
    double tp = ((t2-t1)/getTickFrequency());

    Mat int_pred; test_pred.convertTo(int_pred, CV_32S);
    float pos = countNonZero(int_pred == test_labels);
    float all = test_labels.total();
    float acc = pos / all;
    cerr << format("%-20s %1.4f (%4d/%-4d) accuracy,  (%2.4f/%-2.4f) seconds, ",meth.c_str(), acc, int(pos), int(all), tt, tp);
    cerr << train_data.size() << " train, " << test_data.size()  << " test elems, ";
    cerr << T << "/" << getNumThreads() << " threads used." << endl;
    return 0;
}

