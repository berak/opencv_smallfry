//
// realtime face recognition/transfer learning demo (using facenet features, and an SVM).
// you'll need the pretrained facenet dnn model from here (30.1mb):
//    https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7
//

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/filesystem.hpp>

using namespace std;
using namespace cv;


// default face size, 96 - as required by the pretrained facenet
const int  FIXED_FACE = 96;

// program states
enum
{
    NEUTRAL = 0,
    RECORD  = 1,
    PREDICT = 2
};

class FaceRec
{
    dnn::Net net;
    Ptr<ml::SVM> svm;
    vector<String> names;
public:
    FaceRec(const String &pathToTorchnet)
        : svm(ml::SVM::create())
    {
        svm->setKernel(ml::SVM::LINEAR);
        try {
            net = dnn::readNetFromTorch(pathToTorchnet);
        } catch(Exception &e) {
            cerr << "Download it from:  ";
            cerr << "https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7" << endl;
        }
    }

    Mat process(const Mat &image)
    {
        Mat inputBlob = dnn::blobFromImage(image, 1./255, Size(FIXED_FACE, FIXED_FACE), Scalar(), true, false);
        net.setInput(inputBlob);
        return net.forward();
    }

    int train(const String &imgdir)
    {
        names.clear();
        Mat features, labels;

        vector<String> vec_cls;
        utils::fs::glob(imgdir,"",vec_cls,false,true);
        if (vec_cls.empty())
            return 0;

        for (size_t label=0; label<vec_cls.size(); label++) {
            vector<String> vec_person;
            glob(vec_cls[label], vec_person, false);
            if (vec_person.empty())
                return 0;
            String name = vec_cls[label].substr(imgdir.size()+1);
            cout  << name << " " << vec_person.size() << " images." << endl;
            names.push_back(name);
            for (size_t i=0; i<vec_person.size(); i++) {
                Mat img=imread(vec_person[i]);
                if (img.empty()) continue;
                features.push_back(process(img));
                labels.push_back(int(label));
            }
        }
        svm->train(features, 0, labels);
        //svm->save("faces.xml.gz");
        return vec_cls.size();
    }

    String predict(const Mat & img)
    {
        if (!svm->isTrained()) // not trained, or only 1 person in db
            return names.size() ? names[0] : "";
        Mat feature = process(img);
        float id = svm->predict(feature);
        return names[int(id)];
    }
};

int main(int argc, const char *argv[])
{
    theRNG().state = getTickCount();
    CommandLineParser parser(argc, argv,
        "{h help    || show this help screen}"
        "{v video   |0| video path (a filename, 0 or 1 to use webcam)}"
        "{d dir     |persons| path to the image folder}"
        "{m model   |openface.nn4.small2.v1.t7| path to the trained network model}"
        "{c cascade |haarcascade_frontalface_alt.xml| path to the haarcascade for face detection}"
    );
    String cp = parser.get<String>("video");
    String imgpath = parser.get<String>("dir");
    String torchModel = parser.get<String>("model");
    String cascade_path = parser.get<String>("cascade");
    if (parser.has("help") || torchModel.empty() || imgpath.empty()|| cascade_path.empty()) {
        parser.printMessage();
        return 0;
    }

    cv::CascadeClassifier cascade(cascade_path);
    if (cascade.empty()) {
        cerr << "Could not load cascade: " << cascade_path  << endl;
        return -1;
    }

    VideoCapture cap;
    if (cp == "0") cap.open(0);
    else if (cp == "1") cap.open(1);
    else cap.open(cp);
    if (! cap.isOpened()) {
        cerr << "Could not open capture: " << cp << endl;
        return -2;
    }

    utils::fs::createDirectory(imgpath);
    FaceRec reco(torchModel);
    int n = reco.train(imgpath);
    cout << "train: " << n << " classes." << endl;

    namedWindow("Face Recognition");
    cout << "press 'r'      to record new persons" << endl;
    cout << "      'space'  to stop recording (then input a name on the console)" << endl;
    cout << "      'p'      to predict" << endl;
    cout << "      'esc'    to quit" << endl;

    vector<Mat> images;
    String caption = "";
    int frameNo = 0;
    int state = NEUTRAL;
    Scalar color[3] = {
        Scalar(30,100,30),
        Scalar(10,10,160),
        Scalar(160,100,10),
    };

    while(true) {
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        // if it gets too slow, try with half the size.
        //pyrDown(frame,frame);

        Mat gray;
        cvtColor(frame, gray, COLOR_RGB2GRAY);

        std::vector<cv::Rect> faces;
        cascade.detectMultiScale(gray, faces, 1.2, 3,
            CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH  ,
            Size(40, 40), Size(300,300));

        if (faces.size() > 0) {
            Rect roi = faces[0];
            if ((state == RECORD ) && (frameNo % 3 == 0)) {
                Mat m;
                resize(frame(roi), m, Size(FIXED_FACE,FIXED_FACE));
                images.push_back(m);
                cout << ".";
            }
            if (state == PREDICT) {
                caption = reco.predict(frame(roi));
                if (caption != "") {
                    putText(frame, caption, Point(roi.x, roi.y+roi.width+13),
                        FONT_HERSHEY_PLAIN, 1.3, color[state], 1, LINE_AA);
                }
            }
            rectangle(frame, roi, color[state]);
        }
        for(int i=6,sc=6; i>1; i--,sc+=2) // status led
            circle(frame, Point(10,10), i, color[state]*(float(sc)/10), -1, LINE_AA);

        imshow("Face Recognition", frame);
        int k = waitKey(30);
        if (k == 27 ) break;
        if (k == 'p') state = PREDICT;
        if (k == 'r' && state != RECORD ) {
            images.clear();
            state = RECORD ;
        }
        if (k==' ') {
            if ((state == RECORD ) && (!images.empty())) {
                // ask for a name, and write the images to that folder:
                cout << endl << "please enter a name (leave empty to ignore) :" << endl;
                string n; cin >> n;
                if ((!n.empty()) && (images.size() > 0)) {
                    String folder(imgpath + String("/") + String(n));
                    utils::fs::createDirectory(folder);
                    for (size_t i=0; i<images.size(); i++) {
                        imwrite(format("%s/%6d.jpg", folder.c_str(), theRNG().uniform(0,100000)), images[i]);
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
