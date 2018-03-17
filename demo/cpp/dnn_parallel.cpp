#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/trace.hpp>


using namespace cv;
using std::vector;


/*
OUCH. don't do that !
use EITHER tbb's parallel_for OR opencv's parallel_for_
but never BOTH (or worse, mix)

#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
using namespace tbb;
*/


String modelTxt = "caffenet_deploy_2.prototxt";
String modelBin = "caffe_model_2_iter_15000.caffemodel";

class Parallel_process : public ParallelLoopBody
{
private:

    vector<int> &classIds; // note the reference
    vector<double> &classProbs;
    vector<String> fnClass;

    //
    // all other member vars removed. again, you're not allowed to share them between threads !
    // ( those should be all local )
    //

public:

    Parallel_process(vector<String>& fn, vector<int> &Ids, vector<double> &probs)
        : fnClass(fn), classIds(Ids), classProbs(probs) {}

    void operator()(const cv::Range& range) const
    {
        for(int y = range.start; y < range.end; y++) // start and end are Range member vars, not functions
        {
            Mat inputIm = imread(fnClass[y]);
            if (!inputIm.empty())
            	continue;  // check !

            // you will need one Net per thread (while the sequential version could reuse it),
            // and that's the ... part about this idea.
            dnn::Net net = dnn::readNetFromCaffe(modelTxt, modelBin);
            Mat inputBlob = dnn::blobFromImage(inputIm, 1.0f, Size(227, 227),
                                          Scalar(104, 117, 123),false);

            net.setInput(inputBlob, "data");
            Mat prob = net.forward("prob");  // this has to be a local var, can't be shared between threads !

    		Point classNumber;
            double clsprob;
    		minMaxLoc(prob.reshape(1, 1), NULL, &clsprob, NULL, &classNumber);

            classIds.push_back(classNumber.x);  // so we can retrieve them later
            classProbs.push_back(clsprob);
        }
    }
};


int main(int argc, char **argv)
{
    CV_TRACE_FUNCTION();

    String path("Images/*.png");
    vector<String> fn;
    glob(path,fn,true);
    if (fn.size()==0)
    	return -1; // no images in path

    vector<int> classId_array;
    vector<double> classProb_array;

    // please do not mix tbb and opencv stuff here !
    parallel_for_(Range(0,fn.size()), Parallel_process(fn, classId_array, classProb_array));

	// process  classId_array and   classProb_array

    return 0;
}




/*
int main(int argc, const char* argv[]) {
 	namedWindow("select");
	Mat im = imread("scene.png");
	Rect r = selectROI("select", im);
	cerr << r << endl;
	waitKey();
    return 0;
}
*/
