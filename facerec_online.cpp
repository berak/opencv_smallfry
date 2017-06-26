//
// realtime online demo.
// instead of using/parsing a csv file,
// we save each person to a seperate directory,
// and later parse the glob() output.
//
// YOU NEED AN EMPTY "persons" FOLDER HERE!
// (train images will get saves there,
//   on startup, the app will try to train from those images)
//
// use : online [capture id or path] [img_path] [cascade_path]
//
//

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

using namespace std;
using namespace cv;

#ifdef _WIN32
 const char SEP = '\\';
#else
 const char SEP = '/';
#endif


//
// 3 state online face db
//
enum
{
    NEUTRAL = 0,
    CAPTURE = 1,
    PREDICT = 2
};

//
// default face size, [80..160]
//
enum
{
    FIXED_FACE = 90
};



class FaceRec
{
    Ptr<face::FaceRecognizer>  rec;

public:
    FaceRec()
        : rec(face::LBPHFaceRecognizer::create())
    {}

    Mat preproc(const Mat &img) {
        Mat gray, rez;
        if (img.channels()>1)
            cvtColor(img, gray, COLOR_BGR2GRAY);
        else gray = img;
        const Size cs(FIXED_FACE, FIXED_FACE);
        if (gray.size() != cs) {
            resize(gray, rez, cs);
            return rez;
        }
        return gray;
    }
    int train(const String &imgdir) {
        string last_n("");
        int label(-1);
        map<int,int> classes;
        vector<Mat> features;
        vector<int> labels;

        vector<String> vec;
        glob(imgdir,vec,true);
        if ( vec.empty())
            return 0;

        for (size_t i=0; i<vec.size(); i++) {
            // extract name from filepath:
            String v = vec[i];
            int r1 = v.find_last_of(SEP);
            String v2 = v.substr(0,r1);
            int r2 = v2.find_last_of(SEP);
            String n = v2.substr(r2+1);
            if (n!=last_n) {
                last_n=n;
                label++;
            }
            rec->setLabelInfo(label, n);

            // process img & add to trainset:
            Mat img=imread(vec[i],0);
            features.push_back(preproc(img));
            labels.push_back(label);
            if (classes.find(label)==classes.end())
                classes[label] = 1;
            else
                classes[label] ++;
        }
        rec->train(features, labels);
        for (auto it=classes.begin(); it!=classes.end(); ++it) {
            String name = rec->getLabelInfo(it->first);
            cerr << format("%25s %4d", name.c_str(), it->second) << " images." << endl;
        }
        return classes.size();
    }

    String predict(const Mat & img) {
        int id=-1;
        double dist=-1;
        rec->predict(preproc(img), id, dist);
        if (id < 0)
            return "";

        float conf = 1.0f - sqrt(dist*dist)/250;
        String name = rec->getLabelInfo(id);
        return format("%s : %2.3f", (dist<90?name.c_str():"unknown"), dist);
    }

    bool load(const String &fn) {
        rec = Algorithm::load<face::LBPHFaceRecognizer>(fn);
        return 1;
    }
    bool save(const String &fn) {
        rec->save(fn);
        return 1;
    }
};

int main(int argc, const char *argv[])
{
    theRNG().state = getTickCount();

    cerr << "press 'c' to record new persons," << endl;
    cerr << "      space, to stop recording. (then input a name)." << endl;
    cerr << "      'p' to predict," << endl;
    cerr << "      'n' for neutral," << endl;
    cerr << "      's' to save the current model," << endl;
    cerr << "      esc to quit." << endl;

    string cp("0");
    if (argc > 1) cp=argv[1];

    string imgpath("data\\persons");
    if (argc > 2) imgpath=argv[2];

    std::string cascade_path("haarcascade_frontalface_alt.xml");
    if (argc > 3) cascade_path=argv[3];

    if (argc == 1) {
        cerr << "please use : online [capture id or path] [img_path] [cascade_path]" << endl;
        cerr << "[current]  : online " << cp << " " << imgpath <<  " " << cascade_path << endl << endl;
        return 0;
    }

    namedWindow("reco");

    cv::CascadeClassifier cascade;
    bool clod = cascade.load(cascade_path);
    cerr << "cascade: " << clod  << endl;;

    VideoCapture cap;
    if (cp == "0") cap.open(0);
    else if (cp == "1") cap.open(1);
    else cap.open(cp);
    cerr << "capture(" << cp << ") : " << cap.isOpened() << endl;

    FaceRec reco;
    int n = reco.train(imgpath);
    cerr << "train: " << n << " classes." << endl;

    String save_model = "face.yml.gz";
    // alternatively, load a serialized model.
    //reco.load(save_model);

    vector<Mat> images;
    String caption = "";
    int showCaption = 0;
    int frameNo = 0;
    int state = NEUTRAL;
    Scalar color[3] = {
        Scalar(30,100,30),
        Scalar(10,10,160),
        Scalar(160,100,10),
    };

    while(cap.isOpened() && clod) {
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        // if it gets too slow, try with half the size.
        //pyrDown(frame,frame);

        if (state == PREDICT || state == CAPTURE) {
            Mat gray;
            cvtColor(frame, gray, COLOR_RGB2GRAY);

            std::vector<cv::Rect> faces;
            cascade.detectMultiScale(gray, faces, 1.2, 3,
                CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH  ,
                Size(40, 40), Size(300,300));

            if (faces.size() > 0) {
                Rect roi = faces[0];
                if ((state == CAPTURE) && (frameNo % 3 == 0)) {
                    Mat m;
                    resize(gray(roi), m, Size(FIXED_FACE,FIXED_FACE));
                    images.push_back(m);
                    cerr << ".";
                }
                if (state == PREDICT) {
                    caption = reco.predict(gray(roi));
                    if (caption != "") {
                        putText(frame, caption, Point(roi.x, roi.y+roi.width+13),
                            FONT_HERSHEY_PLAIN, 1.2, color[state], 1, LINE_AA);
                    }
                }
                rectangle(frame, roi, color[state]);
            }
        }
        for(int i=6,sc=6; i>1; i--,sc+=2) // status led
            circle(frame, Point(10,10), i, color[state]*(float(sc)/10), -1, LINE_AA);

        imshow("reco",frame);
        int k = waitKey(30);
        if (k == 27 ) break;
        if (k == 'p') state=PREDICT;
        if (k == 'c' && state != CAPTURE) {
            images.clear();
            state = CAPTURE;
        }
        if (k == 's') {
            cerr << "saved " << save_model << " : " << reco.save(save_model) << endl;
        }
        if (k =='n' || k==' ') {
            if ((state == CAPTURE) && (!images.empty())) {
                // query for a name, and write the images to that folder:
                cerr << endl << "please enter a name(empty to abort) :" << endl;
                char n[200];
                gets(n);
                if ((n[0] != 0) && (images.size() > 0)) {
                    String path = imgpath;
                    path += SEP;
                    path += n;
                    String cmdline = "mkdir ";
                    cmdline += path;
                    cerr << cmdline << endl;
                    system(cmdline.c_str()); // lame, but portable..
                    for (size_t i=0; i<images.size(); i++) {
                        imwrite(format("%s/%6d.png", path.c_str(), theRNG().next()), images[i]);
                    }
                    reco.train(imgpath);
                }
            }
            state = NEUTRAL;
        }
        frameNo++;
    }
    return 0;
}
