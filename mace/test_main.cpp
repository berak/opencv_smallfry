#include "opencv2/ts.hpp"
#include "opencv2/objdetect.hpp"
#include "mace.h"



using namespace cv;
using namespace testing;
using namespace std;

#define TESTSET_NAMES testing::Values("david","dudek","faceocc2")

const string TRACKING_DIR = "tracking";
const string FOLDER_IMG = "data";
const string FOLDER_OMIT_INIT = "initOmit";


class MaceTest
{
 public:

  MaceTest(string _video, int salt, bool multi);
  virtual ~MaceTest() {}
  virtual void run();

 protected:
  std::vector<std::string> splitString( std::string s, std::string delimiter );
  vector<Mat> samples(const string &name, int N,int off=0);
  int found(const string &vid);

  Ptr<MACE> mace;
  CascadeClassifier cascade;
  string video;
  string vidA, vidB;
  int startFrame;
  int nSampsTest;
  int nSampsTrain;
  int nStep;
};

MaceTest::MaceTest(string _video, int salt, bool multi)
{
	  int Z = 64; // window size
    if (multi)
        mace = MACE::createSampler(Z,vector<Rect2d>{
            Rect2d(0,0,1,1),           // whole image
            Rect2d(0.25,0.5,0.5,0.5),  // bottom center(mouth)
            Rect2d(0,0,0.5,0.5),       // top left (eye)
            Rect2d(0.5,0,0.5,0.5)      // top right (eye)
        });
    else
        mace = MACE::create(Z);
    if (salt)
        mace->salt(salt);
    video = _video;
    if (video=="david") { vidA="dudek"; vidB="faceocc2"; }
    if (video=="dudek") { vidA="david"; vidB="faceocc2"; }
    if (video=="faceocc2") { vidA="dudek"; vidB="david"; }
    cerr << "created test for " << video << endl;

    startFrame = 600;
    nStep = 2;
    nSampsTest = 5;
    nSampsTrain = 25;

    cascade.load(cvtest::TS::ptr()->get_data_path() + "/cascadeandhog/cascades/haarcascade_frontalface_alt.xml");
    CV_Assert(!cascade.empty());
}

std::vector<std::string> MaceTest::splitString( std::string s, std::string delimiter )
{
  std::vector<string> token;
  size_t pos = 0;
  while ( ( pos = s.find( delimiter ) ) != std::string::npos )
  {
    token.push_back( s.substr( 0, pos ) );
    s.erase( 0, pos + delimiter.length() );
  }
  token.push_back( s );
  return token;
}

void MaceTest::run() {
  cerr << "running test for " << video << endl;
  vector<Mat> sam_train = samples(video, nSampsTrain, 0);
  mace->train(sam_train);
  int self_ok = found(video);
  int false_A = found(vidA);
  int false_B = found(vidB);
  ASSERT_EQ(self_ok, nSampsTest);
  ASSERT_EQ(false_A, 0);
  ASSERT_EQ(false_B, 0);
}

int MaceTest::found(const string &vid) {
  vector<Mat> sam_test = samples(vid, nSampsTest, (1+2*nSampsTrain));
  int hits = 0;
  for (size_t i=0; i<sam_test.size(); i++) {
    hits += mace->same(sam_test[i]);
  }
  cout << vid << " " << hits << " " << sam_test.size() << endl;
  return hits;
}

vector<Mat> MaceTest::samples(const string &name, int N, int off) {
  string folder = cvtest::TS::ptr()->get_data_path() + "/" + TRACKING_DIR + "/" + name + "/" + FOLDER_IMG;
  string vid =  folder + "/" + name + ".webm" ;
  VideoCapture c;
  c.open(vid);
  vector<Mat> samps;
  while (samps.size() < N) {
    int frameNo = startFrame + off;
    c.set(CAP_PROP_POS_FRAMES, frameNo);
    //cout << frameNo << " " << c.get(CAP_PROP_FRAME_COUNT) << " " << N << " " << samps.size() << endl;
    Mat frame;
    vector<Rect> rects;
    c >> frame;
    CV_Assert(!frame.empty());
    cascade.detectMultiScale(frame,rects);
    off += nStep;
    if (rects.empty())
      continue;
    samps.push_back(frame(rects[0]));
    /*rectangle(frame,rects[0],Scalar(200,0,0));
    imshow("W", frame);
    waitKey(3);*/
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


TEST_P(MACE_, unsalted_single)
{
  MaceTest test(dataset, 0, false); test.run();
}
TEST_P(MACE_, unsalted_multi)
{
  MaceTest test(dataset, 0, true); test.run();
}
TEST_P(MACE_, salted_single)
{
  MaceTest test(dataset, 627364238, false); test.run();
}
TEST_P(MACE_, salted_multi)
{
  MaceTest test(dataset, 627364238, true); test.run();
}


INSTANTIATE_TEST_CASE_P( Face, MACE_, TESTSET_NAMES);

CV_TEST_MAIN("cv")
