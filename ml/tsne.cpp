#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

//
//
// http://lvdmaaten.github.io/tsne/
//
// code here mostly adapted from:
// https://github.com/karpathy/tsnejs
//
//
class TSNE
{
    // helper function
    static int sign(float x) { return x > 0 ? 1 : x < 0 ? -1 : 0; }

    // compute pairwise distance in all vectors in X
    Mat xtod(const Mat &X) const
    {
        Mat_<float> dist = Mat_<float>::zeros(N,N);
        for(int i=0; i<N; i++)
        {
            for(int j=i+1; j<N; j++)
            {
                dist(i,j) = dist(j,i) = (float)norm(X.row(i), X.row(j), NORM_L2SQR);
            }
        }
        //cerr << "D " << dist << endl;;
        return dist;
    }

    // compute (p_{i|j} + p_{j|i})/(2n)
    Mat d2p(const Mat &_D, float perplexity, float tol) const
    {
        Mat_<float> D(_D);               // for easy () access
        //CV_Assert(D.rows == D.cols);     // D should have square number of elements
        float Htarget = log(perplexity); // target entropy of distribution
        cerr << "Htarget " << Htarget << endl;
        Mat_<float> p;
        for(int i=0; i<N; i++)
        {
            float betamin = -FLT_MAX;
            float betamax = FLT_MAX;
            float beta = 1; // initial value of precision
            bool done = false;
            int maxtries = 50;

            // perform binary search to find a suitable precision beta
            // so that the entropy of the distribution is appropriate
            int numtries = 0;
            Mat_<float> prow(1,N);
            while(!done)
            {
                // compute entropy and kernel row with beta precision
                float psum = 0.0f;
                for(int j=0; j<N; j++)
                {
                    // we dont care about diagonals
                    float pj = (i == j) ? 0 : exp(-D(i,j) * beta);
                    prow(j) = pj;
                    psum += pj;
                }
                // normalize p and compute entropy
                float Hhere = 0.0f;
                for(int j=0; j<N; j++)
                {
                    float pj = prow(j) / psum;
                    prow(j) = pj;
                    if (pj > 1e-7f) Hhere -= pj * log(pj);
                }

                // adjust beta based on result
                if(Hhere > Htarget)
                {
                    // entropy was too high (distribution too diffuse)
                    // so we need to increase the precision for more peaky distribution
                    betamin = beta; // move up the bounds
                    beta = (betamax == FLT_MAX) ? beta * 2 : (beta + betamax) / 2;
                }
                else
                {
                    // converse case. make distrubtion less peaky
                    betamax = beta;
                    beta = (betamin == -FLT_MAX) ? beta / 2 : (beta + betamin) / 2;
                }
                //cerr << numtries << " Hhere " << Hhere << " beta " << beta << endl;

                // stopping conditions: too many tries or got a good precision
                done = ((abs(Hhere - Htarget) < tol) || (++numtries >= maxtries));
            }
            p.push_back(prow);
        } // end loop over examples i

        // symmetrize P and normalize it to sum to 1 over all ij
        Mat_<float> Pout(N,N);
        float N2(N*2);
        for(int i=0; i<N; i++)
        {
            for(int j=0; j<N; j++)
            {
                Pout(i,j) = std::max((p(i,j) + p(j,i))/N2, 1e-10f);
            }
        }
        //cerr << "P " << Pout << endl;;
        return Pout;
    }


    // return cost and gradient, given an arrangement
    void costGrad(Mat_<float> &grad, float &cost)
    {
        int pmul = iter < 100 ? 4 : 1; // trick that helps with local optima

        // compute current Q distribution, unnormalized first
        Mat_<float> Qu = Mat_<float>::zeros(N,N);
        float qsum = 0.0f;
        for(int i=0; i<N; i++)
        {
            for(int j=i+1; j<N; j++)
            {
                float dsum = 0.0f;
                for(int d=0; d<dim; d++)
                {
                    float dhere = Y(i,d) - Y(j,d);
                    dsum += dhere * dhere;
                }
                float qu = 1.0f / (1.0f + dsum); // Student t-distribution
                Qu(i,j) = Qu(j,i) = qu;
                qsum += 2 * qu;
            }
        }
        // normalize Q distribution to sum to 1
        Mat_<float> Q = cv::max(Qu / qsum, 1e-10);

        cost = 0.0f;
        grad = Mat_<float>::zeros(Y.size()); // init grad for point i
        for(int i=0; i<N; i++)
        {
            for(int j=0; j<N; j++)
            {
                // accumulate cost (the non-constant portion at least...)
                cost += - P(i,j) * log(Q(i,j)); /// epsilon added to avoid INF
                float premult = 4 * (pmul * P(i,j) - Q(i,j)) * Qu(i,j);
                for(int d=0; d<dim; d++)
                {
                    grad(i,d) += premult * (Y(i,d) - Y(j,d));
                }
            }
        }
    }

public:

    TSNE(const Mat &X, int dim=2, float perplexity=30, float epsilon=10)
        : N(X.rows)
        , perplexity(perplexity)
        , epsilon(epsilon)
        , dim(dim)
        , iter(0)
        , Y(Mat_<float>::ones(X.rows,dim))
        , gains(Mat_<float>::ones(X.rows,dim))// step gains to accelerate progress in unchanging directions
        , ystep(Mat_<float>::zeros(X.rows,dim))// momentum accumulator
    {
        Mat dists = X;//xtod(X); // convert X to distances using gaussian kernel
        P = d2p(dists, perplexity, 1e-4f); // attach to object
        // generate random solution to t-SNE
        //randu(Y, -1e-4f, 1e-4f);
        randn(Y, 0, 1e-4f);
        //cerr << "D " << dists << endl;
        //cerr << "P " << P << endl;
        //cerr << "Y " << Y << endl;
    }

    // perform a single step of optimization to improve the embedding
    float step()
    {
        iter += 1;

        float cost;
        Mat_<float> grad;
        costGrad(grad, cost); // evaluate gradient

        // perform gradient step
        Mat_<float> ymean = Mat_<float>::zeros(1, dim);
        for(int i=0; i<N; i++)
        {
            for(int d=0; d<dim; d++)
            {
                float gid = grad(i,d);
                float sid = ystep(i,d);
                float gainid = gains(i,d);

                // compute gain update
                float newgain = sign(gid) == sign(sid) ? gainid * 0.8f : gainid + 0.2f;
                if (gainid < 0.01f) gainid = 0.01f; // clamp
                gains(i,d) = newgain; // store for next turn

                // compute momentum step direction
                float momval = iter < 250 ? 0.5f : 0.8f;
                float newsid = momval * sid - epsilon * newgain * grad(i,d);
                ystep(i,d) = newsid; // remember the step we took

                // step!
                Y(i,d) += newsid;

                ymean(d) += Y(i,d); // accumulate mean so that we can center later
            }
        }

        // reproject Y to be zero mean
        for(int i=0; i<N; i++)
        {
            for(int d=0; d<dim; d++)
            {
                Y(i,d) -= ymean(d)/N;
            }
        }

        //cerr << "Y " << Y << endl;;
        if(iter%50==0) cerr << "iter " << iter << ", cost: " << cost << " grad: " << (sum(grad)[0]) << endl ;//<< " grad:" << grad <<  endl;
        return cost; // return current cost
    }

    Mat compute(int steps)
    {
        while(iter < steps)
        {
            step();
        }
        return Y;
    }

    Mat_<float> P;
    Mat_<float> Y;
    Mat_<float> gains,ystep;
    float perplexity;
    float epsilon;
    int iter, dim, N;
};

int simple()
{
    Mat_<float> X(3,3);
    //X << 1.0, 0.1, 0.2,
    //     0.1, 1.0, 0.3,
    //     0.2, 0.1, 1.0;
    X <<  0,0.1,0.2,0.1,0,0.3,0.2,0.3,0;
    TSNE tsne(X, 2, 30);
    Mat Y = tsne.compute(500);
    cerr << Y << endl;;
    return 0;
}

int main( int argc, const char** argv )
{
    //return simple();

    int N = 100;
    int S = 24;
    int T = 35;
    int IT = 4500;
    int PERP = 20;
    float EPS = 0.027;
    int SCALE = 180;
    //String path = "e:/media/faces/tv/*.png";
    String path = "c:/data/faces/att/*.pgm";
    vector<String> fn;
    glob(path,fn,true);
    Mat data;
    for (size_t i=0; i<N; i++)
    {
        Mat im = imread(fn[i],0);
        resize(im,im,Size(S,S));
        im.convertTo(im,CV_32F,1.0f/255);
        data.push_back(im.reshape(1,1));
    }
    TSNE tsne(data, 2, PERP, EPS);
    //Mat_<float> Y = tsne.compute(IT);

    while(tsne.iter < IT)
    {
        float cost = tsne.step();
        Mat_<float> &Y = tsne.Y;
        if (tsne.iter%20 != 0) continue;

        Mat draw(500,500,CV_8UC1, Scalar::all(0));
        Rect bounds(0,0,500,500);
        for (int i=0; i<Y.rows; i++)
        {
            Mat im = data.row(i).reshape(1,S);
            Mat im2; resize(im,im2,Size(T,T));
            im2.convertTo(im2,draw.type(),255);

            Point p(250 + Y(i,0)*SCALE, 250 + Y(i,1)*SCALE);
            //Point p(abs(Y(i,0))*SCALE*2, abs(Y(i,1))*SCALE*2);
            Rect r(p.x-T/2, p.y-T/2, T,T);
            Rect o = r & bounds;
            if (o.area()<r.area())
                continue;
            im2.copyTo(draw(r));
        }
        imshow("o2",draw);
        waitKey(20);
    }
    cerr << "finished!" << endl;
    waitKey();
    return 0;
}
