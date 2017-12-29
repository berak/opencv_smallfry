#include "opencv2/ts.hpp"
#include "mace.h"


using namespace cv;
using namespace testing;
using namespace std;

//
// try to test on one person, and test against the other 2
//
#define TESTSET_NAMES testing::Values("david","dudek","faceocc2")

const string TRACKING_DIR = "tracking";
const string FOLDER_IMG = "data";


class MaceTest
{
public:

    MaceTest(string _video, bool salt);
    void run();

protected:
    vector<Rect> boxes(const string &fn);
    vector<Mat> samples(const string &name, int N,int off=0);
    int found(const string &vid);

    Ptr<MACE> mace;

    string video;      // train
    string vidA, vidB; // test

    int nSampsTest;
    int nSampsTrain;
    int nStep;
    bool salt;
};

MaceTest::MaceTest(string _video, bool salt)
{
    int Z = 64; // window size
    mace = MACE::create(Z);

    video = _video;
    if (video=="david") { vidA="dudek"; vidB="faceocc2"; }
    if (video=="dudek") { vidA="david"; vidB="faceocc2"; }
    if (video=="faceocc2") { vidA="dudek"; vidB="david"; }

    nStep = 2;
    nSampsTest = 5;
    nSampsTrain = 35;
    this->salt = salt;
}

vector<Rect> MaceTest::boxes(const string &fn)
{
    ifstream in(fn);
    int x,y,w,h;
    char sep;
    vector<Rect> boxes;
    while (in.good() && (in >> x >> sep >> y >> sep >> w >> sep >> h))
    {
        boxes.push_back( Rect(x,y,w,h) );
    }
    return boxes;
}

void MaceTest::run() {
    vector<Mat> sam_train = samples(video, nSampsTrain, 0);
    if (salt) mace->salt(video); // "owner's" salt with "two factor"
    mace->train(sam_train);
    int self_ok = found(video);
    if (salt) mace->salt(vidA); // "other's" salt
    int false_A = found(vidA);
    if (salt) mace->salt(vidB); // "other's" salt
    int false_B = found(vidB);
    ASSERT_GE(self_ok, nSampsTest/2);  // it may miss positives
    ASSERT_EQ(false_A, 0);  // but *absolutely* no false positives allowed.
    ASSERT_EQ(false_B, 0);
}

int MaceTest::found(const string &vid) {
    vector<Mat> sam_test = samples(vid, nSampsTest, (1+nStep*nSampsTrain));
    int hits = 0;
    for (size_t i=0; i<sam_test.size(); i++) {
        hits += mace->same(sam_test[i]);
    }
    cout << vid << " " << hits << " " << sam_test.size() << endl;
    return hits;
}

vector<Mat> MaceTest::samples(const string &name, int N, int off) {
    string folder = cvtest::TS::ptr()->get_data_path() + TRACKING_DIR + "/" + name;
    string vid  = folder + "/" + FOLDER_IMG + "/" + name + ".webm";
    string anno = folder + "/gt.txt";
    vector<Rect> bb = boxes(anno);
    int startFrame = (name=="david") ? 300 : 0;
    VideoCapture c;
    bool ok = c.open(vid);
    vector<Mat> samps;
    while (samps.size() < N) {
        int frameNo = startFrame + off;
        c.set(CAP_PROP_POS_FRAMES, frameNo);
        Mat frame;
        c >> frame;
        Rect r = bb[off];
        off += nStep;
        samps.push_back(frame(r));
      //  rectangle(frame,r,Scalar(200,0,0));
      //  imshow("W", frame);
      //  waitKey(30);
    }
    c.release();
    return samps;
}

//[TESTDATA]
PARAM_TEST_CASE(MACE_, string)
{
    string dataset;
    virtual void SetUp()
    {
        dataset = GET_PARAM(0);
    }
};


TEST_P(MACE_, unsalted)
{
    MaceTest test(dataset, false); test.run();
}
TEST_P(MACE_, salted)
{
    MaceTest test(dataset, true); test.run();
}


INSTANTIATE_TEST_CASE_P(Face, MACE_, TESTSET_NAMES);

CV_TEST_MAIN("cv")
