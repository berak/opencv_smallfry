#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
using namespace cv;

#include <algorithm>
#include <iostream>
#include <set>
using namespace std;

struct Cluster
{
    virtual bool cluster(const Mat &in, Mat &indices, int K=0) const = 0;

    static void vecMat(const Mat &in, vector<Mat> &feats)
    {
        for (int i=0; i<in.rows; i++)
        {
            feats.push_back(in.row(i));
        }
    }
};



struct Partition : Cluster
{
    struct Dist
    {
        double D;
        Dist(double d) : D(d) {}
        bool operator()(const Mat &a, const Mat &b) const
        {
            return norm(a,b) < D;
        }
    };

    bool cluster(const Mat &in, Mat &indices, int K=0) const
    {
        vector<Mat> feats;
        vecMat(in, feats);

        // learn a distance
        double d = 0.5;
        vector<int> labels;
        for(int j=0; j<500; j++)
        {
            // partition
            labels.clear();
            cv::partition(feats,labels,Dist(d));

            // count uniques
            std::set<int> classes;
            for (size_t i=0; i<labels.size(); i++)
                classes.insert(labels[i]);
            int n = classes.size();

            // loss
            int D = n - K;
            double step = 1.0 + double(D)/((j/4+1)*labels.size()); // adaptive learnrate
            d *= step;
            cerr << j << "\tn " << n << "\tD " << D << "\td " << d << "\ts " << step <<  endl;
            if (D == 0) break;
        }
        Mat(labels).copyTo(indices);
        return true;
    }
    static Ptr<Cluster> create() {return makePtr<Partition>();}
};


struct Em : Cluster
{
    bool cluster(const Mat &in, Mat &indices, int K=0) const
    {
        Ptr<ml::EM> em = ml::EM::create();
        if (K!=0)
            em->setClustersNumber(K);
        return em->trainEM(in, noArray(), indices);
    }
    static Ptr<Cluster> create() {return makePtr<Em>();}
};


struct KMeans : Cluster
{
    bool cluster(const Mat &in, Mat &indices, int K=0) const
    {
        kmeans(in,K,indices,TermCriteria(),3,KMEANS_PP_CENTERS);
        return false;
    }
    static Ptr<Cluster> create() {return makePtr<KMeans>();}
};



int main(int argc, char* argv[])
{
    const int C=10;  // num classes
    const int I=10;  // instances per class
    const int N=I*C; // img count
    const int S=24;  // img size
    String meth = "Part";
    if (argc>1) meth = argv[1];

    String path = "c:/data/faces/att/";
    //String path = "e:/media/faces/tv10/";
    vector<String> fn;
    glob(path,fn,true);
    Mat data;
    for (size_t i=0; i<N; i++)
    {
        Mat im = imread(fn[i],0);
        if (im.empty())
        {
            cerr << "invalid " << fn[i] << endl;
            continue;
        }
        resize(im,im,Size(S,S));
        im.convertTo(im,CV_32F,1.0f/255);
        data.push_back(im.reshape(1,1));
    }
    cerr << meth << " " << data.size() << " / " << (data.total()*data.elemSize()) << endl;

    //PCA pca(data,Mat(),0);
    //Mat proj = pca.project(in);
    //cerr << "proj: " << proj.size() << endl;

    Ptr<Cluster> group;
    if (meth.substr(0,2)=="Em")   group = Em::create();
    if (meth.substr(0,2)=="KM")   group = KMeans::create();
    if (meth.substr(0,4)=="Part") group = Partition::create();

    Mat_<int> ind(1, N, 0);
    Mat_<int> cls(1, C, 0);
    int64 t0 = getTickCount();
    group->cluster(data, ind, C);
    int64 t1 = getTickCount();

    float err = 0.0000001f;
    for (int j=0; j<C; j++)
    {
        // majority vote:
        Mat_<int> votes(1,C,0);
        for (int i=0; i<I; i++)
        {
            int idx = std::min(ind(j*I+i), (C-1));
            votes(idx) ++;
        }
        double me=0;
        minMaxIdx(votes, 0, &me, 0, 0);
        cls(j) += int(me);

        float loss = float(I - me);
        err += loss;//*loss;
        //cerr << votes << endl;
    }
    //cerr << ind.reshape(1,I) << endl;
    cerr << "cls " << cls << endl;
    float errn = (err);
    float acc = 1.0f - errn/N;
    cerr << "err: " << errn << " / " << N << "  acc: " << acc << " time: " << ((t1-t0)/getTickFrequency()) << endl;

    namedWindow("clust",0);
    for (int k=0; k<C; k++)
    {
        Mat draw(S*4, S*4, data.type(), Scalar(0));
        Rect r(0,0,S,S);
        for (int i=0; i<ind.total(); i++)
        {
            if (ind(i) != (k))
                continue;
            data.row(i).reshape(1,S).copyTo(draw(r));
            r.x += S;
            if (r.x >= draw.cols)
            {
                r.x=0;
                r.y+=S;
                if (r.y >= draw.rows)
                    break;
            }
        }
        imshow("clust",draw);
        waitKey();
    }
    return EXIT_SUCCESS;
}

