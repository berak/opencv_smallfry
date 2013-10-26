#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>

using namespace cv;

class ConDensation 
{
public:

    //
    //! All Matrices interfaced here are expected to have dp cols, and are float.
    //
    ConDensation( int dp, int numSamples, float flocking=0.9f );

    //! Reset. call at least once before correct()
    void initSampleSet( const Mat & lowerBound, const Mat & upperBound, const Mat & dyna=Mat() );

    //! Update the state and return prediction.
    const Mat & correct( const Mat & measurement );

    //! Access single samples(read only).
    int   sampleCount()       { return samples.rows; }
    float sample(int j,int i) { return samples(j,i); }

private:

    int DP;                      //! Sample dimension
    int numSamples;              //! Number of the Samples                 
    float flocking;              //! flocking/congealing factor
    Mat_<float> range;           //! Scaling factor for correction, the upper bound from the orig. samples
    Mat_<float> dynamMatr;       //! Matrix of the linear Dynamics system  
    Mat_<float> samples;         //! Arr of the Sample Vectors             
    Mat_<float> newSamples;      //! Temporary array of the Sample Vectors 
    Mat_<float> confidence;      //! Confidence for each Sample            
    Mat_<float> cumulative;      //! Cumulative confidence                 
    Mat_<float> randomSample;    //! RandomVector to update sample set     
    Mat_<float> state;           //! Predicted state vector
    Mat_<float> mean;            //! Internal mean vector
    Mat_<float> measure;         //! Cached measurement vector

    struct Rand                  //! CvRandState replacement
    {
        RNG r;
        float lo, hi;
        Rand(float l=0.0f,float h=1.0f) { set(getTickCount(),l,h); }
        void set(int64 s=0, float l=0.0f, float h=1.0f) { r.state=s; lo=l; hi=h;}
        float uni() { return r.uniform(lo,hi); }
    };
    std::vector<Rand> rng;       //! One rng for each dimension.

    void updateByTime();
};


ConDensation::ConDensation( int dp, int numSamples, float flocking )
    : DP(dp)
    , numSamples(numSamples)
    , flocking(flocking)
    , rng(DP)
{
}

void ConDensation::initSampleSet( const Mat &lower, const Mat & upper, const Mat & dyna )
{
    CV_Assert( (lower.type()==CV_32FC1) && (upper.type()==CV_32FC1) );
    CV_Assert( (lower.cols==DP) && (upper.cols==DP) );
    CV_Assert( (lower.rows==1) && (upper.rows==1) );
    Mat_<float>lowerBound(lower);
    Mat_<float>upperBound(upper);
    range = upper; // cache for reuse in correct()

    // dyna might be empty (then we'll use an identity matrix), or a DP x DP x float transformation mat
    CV_Assert( dyna.empty() || ((dyna.rows==DP) && (dyna.rows==DP) && (dyna.type()==CV_32FC1)) );
    dynamMatr    = dyna.empty() ? Mat_<float>::eye(DP,DP) : dyna;

    cumulative   = Mat_<float>::zeros(numSamples,1);
    samples      = Mat_<float>::zeros(numSamples,DP);
    newSamples   = Mat_<float>::zeros(numSamples,DP);
    randomSample = Mat_<float>::zeros(1,DP);
    state        = Mat_<float>::zeros(1,DP);
    mean         = Mat_<float>::zeros(1,DP);
    confidence   = Mat_<float>( numSamples, 1, 1.f / numSamples );


    for( int d=0; d<DP; d++ )
    {
        rng[d].set(getTickCount(),lowerBound(d),upperBound(d));
    }
    // Generating the samples 
    for( int s=0; s<numSamples; s++ )
    {
        for( int d=0; d<DP; d++ )
        {
            samples(s,d) = rng[d].uni();
        }
    }
}


void ConDensation::updateByTime()
{
    // Calculating the Mean 
    mean.setTo(0);
    float sum = 0.0f;
    for( int s=0; s<numSamples; s++ )
    {
        state = samples.row(s) * confidence(s);
        mean += state;
        sum  += confidence(s);
        cumulative(s) = sum;
    }

    // Taking the new state vector from transformation of mean by dynamics matrix 
    mean /= sum;
    state = mean * dynamMatr;
    //sum  /= numSamples;
    // Initialize the random number generator.
    RNG rngUp(getTickCount());

    // We want a record of the span of the particle distribution. 
    // The resampled distribution is dependent on this quantity.
    std::vector<float> sampleMax(DP,FLT_MIN), sampleMin(DP,FLT_MAX);
    // Updating the set of random samples 
    // The algorithm of the original code always picked the last
    // sample, so was not really a weighted random re-sample.  It
    // wasn't really random, either, due to careless seeding of the
    // random number generation.

    // This version resamples according to the weights calculated by
    // the calling program and tries to be more consistent about
    // seeding the random number generator more carefully.
    for( int s=0; s<numSamples; s++ )
    {
	    // Choose a random number between 0 and the sum of the particles' weights.
        float randNumber = rngUp.uniform(0.0f,sum);

	    // Use that random number to choose one of the particles.
        int j = 0;
        while( (cumulative(j) <= randNumber) && (j<numSamples-1))
        //while( (cumulative(j) <= (float) s * sum) && (j<numSamples-1))
        {
            j++;
        }

	    // Keep track of the max and min of the sample particles.
	    // We'll use that to calculate the size of the distribution.
	    for (int d=0; d<DP; d++) 
        {
            newSamples(s,d) = samples(j,d); 
	        sampleMax[d] = max(sampleMax[d], newSamples(s,d));
	        sampleMin[d] = min(sampleMin[d], newSamples(s,d));
	    }
    }

    // Reinitializes the structures to update samples randomly 
    for(int d=0; d<DP; d++)
    {
	    float diff = flocking * (sampleMax[d] - sampleMin[d]);

        if ( 0 )
        {
	        // This line may not be strictly necessary, but it prevents
	        // the particles from congealing into a single particle in the
	        // event of a poor choice of fitness (weighting) function.
	        diff = max(diff, 0.02f * newSamples(0,d));
        } else {
            // Rule 1 : reaching the target is the goal here, right ? 
            // * if we lost it         : swarm out  
            // * if target was reached : hog it .
            diff = min(diff, flocking * (measure(d) - newSamples(0,d)));
        }

        // Re-seed and set the limits to the geometric extent of the distribution.
        rng[d].set(getTickCount()+d,-diff,diff);
        // extra spin on the electronic roulette.(sic)
        rng[d].uni();
    }
    // Adding the random-generated vector to every projected vector in sample set
    for( int s=0; s<numSamples; s++ )
    {
        Mat_<float> r = newSamples.row(s) * dynamMatr;
        for( int d=0; d<DP; d++ )
        {
            samples(s,d) = r(d)+ rng[d].uni();
        }
    }
}

//
//! adjust confidence based on euclidean distance and return predicted state
//
const Mat & ConDensation::correct( const Mat & measurement ) 
{
    measure = measurement;
    for (int s = 0; s < numSamples; s++) 
    {
        double dist = 0;
        for( int d=0; d<DP; d++ )
        {
            float diff = (measure(d) - samples(s,d))/range(d);
            dist += diff*diff;
        }
        confidence(s) = float(exp(-100.0f * sqrt(dist/(DP*DP))));
    }
    updateByTime();
    return state;
}



//
//---8<--------------usage test------------------------------------------------
//


#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace std;
void drawCross( Mat & img, Point center, Scalar color, int d ) {
  line( img, cv::Point( center.x - d, center.y - d ),			
	cv::Point( center.x + d, center.y + d ), color, 2, LINE_AA, 0);	
  line( img, cv::Point( center.x + d, center.y - d ),			
	cv::Point( center.x - d, center.y + d ), color, 2, LINE_AA, 0 );
}

struct mouse_info_struct { 
    int x,y; 
} mouse_info = {-1,-1}, last_mouse;

std::vector<cv::Point> mouseV, particleV;
int counter = -1;

// Define this to proceed one click at a time.
//#define CLICK 1
#define PLOT_PARTICLES 1


void on_mouse(int event, int x, int y, int flags, void* param) {
#ifdef CLICK
    if (event == EVENT_LBUTTONUP) 
#endif
    {
        last_mouse = mouse_info;
        mouse_info.x = x;
        mouse_info.y = y;
        counter = 0;
    }
}

int main (int argc, char * const argv[]) 
{
    int DP = 2;
    int nParticles = 62;
    float xRange = 650.0f;
    float flocking = 0.9f;
    float minRange[] = { 0.0f,0.0f };
    float maxRange[] = { xRange, xRange };
    Mat_<float> LB(1,DP, minRange);
    Mat_<float> UB(1,DP, maxRange);
    Mat_<float> measurement(1,DP); 
    Mat_<float> dyna(Mat_<float>::eye(2,2));

    ConDensation condens(DP, nParticles);

    cv::Mat img(int(xRange), int(xRange), CV_8UC3);
    cv::namedWindow("mouse particle");
    cv::setMouseCallback("mouse particle", on_mouse, 0);

    char code = (char)-1;
    for(;;) 
    {

        if (mouse_info.x < 0 || mouse_info.y < 0) 
        {
            imshow("mouse particle", img);
            cv::waitKey(30);
            continue;
        }

        condens.initSampleSet( LB, UB, dyna );

        mouseV.clear();
        particleV.clear();

        for(;;) 
        {
            code = (char)cv::waitKey(30);
            if( code > 0 )
                break;

            #ifdef CLICK
            if (counter++ > 0) {
                continue;
            } 
            #endif

            measurement(0) = float(mouse_info.x);
            measurement(1) = float(mouse_info.y);

            cv::Point measPt(mouse_info.x,mouse_info.y);
            mouseV.push_back(measPt);

            // Clear screen
            img = cv::Scalar::all(60);

            // Update and get prediction:
            const Mat_<float> & pred = condens.correct(measurement);

            cv::Point2f statePt(pred(0), pred(1));
            particleV.push_back(statePt);

            #ifdef PLOT_PARTICLES
            for (int s=0; s<condens.sampleCount(); s++) {
                cv::Point2f partPt(condens.sample(s,0), condens.sample(s,1));
                drawCross(img, partPt, cv::Scalar(255,90,(int)(s * 255.0/(float)condens.sampleCount())), 2);
            }
            #endif

            for (size_t i = 0; i < mouseV.size() - 1; i++) {
                line(img, mouseV[i], mouseV[i+1], cv::Scalar(255,255,0), 1);
            }
            for (size_t i = 0; i < particleV.size() - 1; i++) {
                line(img, particleV[i], particleV[i+1], cv::Scalar(0,255,0), 1);
            }
            drawCross( img, statePt, cv::Scalar(255,255,255), 5 );
            drawCross( img, measPt,  cv::Scalar(0,0,255), 5 );

            imshow( "mouse particle", img );
        }

        if( code == 27 || code == 'q' || code == 'Q' )
            break;
    }

    return 0;
}
