#include "tiny_dnn/tiny_dnn.h" // order matters, on win, this has to go before opencv (ACCESS_WRITE)
#include "tiny_dnn/util/image.h"
#include "opencv2/opencv.hpp"
#include "opencv2/tracking.hpp"
#include <iostream>
#include <deque>

using namespace cv;
using namespace std;



using namespace tiny_dnn;
using namespace tiny_dnn::layers;
//! load a json model from file,
//!  optionally load pretrained weights
int load_nn(network<sequential> &nn, const string &json_model, const string &pre_weigths="") {
    try {
        nn.load(json_model.c_str(), content_type::model, file_format::json);
        if (! pre_weigths.empty()) {
            cout << "reading weights from " << pre_weigths << endl;
            ifstream ifs(pre_weigths.c_str());
            ifs >> nn;
        }
    } catch (const nn_error& e) {
       std::cout << e.what();
    }
    return 0;
}

int HAVE_NEUTRAL=0;

const char *names[] = {
    "neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"
};
int videoTest(const String &json, const String &weights, int channels) {
    network<sequential> nn;
    load_nn(nn, json, weights);

    Ptr<Tracker> track;

    VideoCapture cap(0);
    CascadeClassifier cad("haarcascade_frontalface_alt.xml");
    deque<Mat> queue;
    while(cap.isOpened()) {
        Mat frame, gray;
        cap >> frame;
        cv::resize(frame,frame,cv::Size(frame.cols/2, frame.rows/2));
        cv::cvtColor(frame,gray,CV_BGR2GRAY);
        cv::equalizeHist(gray,gray);

        Rect box, screen(0, 0, frame.cols, frame.rows);
        Rect2d box2d;
        if (track.empty() || !track->update(gray,box2d)) {
            vector<Rect> faces;
            cad.detectMultiScale(gray,faces,1.1,4,CV_HAAR_FIND_BIGGEST_OBJECT,cv::Size(30,30));
            if (faces.size()) {
                box = faces[0];
                box.height *= 1.05;
                box2d = box;
                track = TrackerMOSSE::create();
                bool ok = track->init(gray, box2d);
                cout << ok << " " <<  box << endl;
            }
        } else {
            box = box2d;
        }
        box &= screen;
        if (! box.empty()) {
            rectangle(frame, box, Scalar(0,200,0), 1);
            Mat det = gray(box);
            resize(det,det,Size(64,64));
            det.convertTo(det,CV_32F);
            Scalar m,d;
            meanStdDev(det,m,d);
            det -= m[0];
            det /= d[0];
            queue.push_back(det);
            if (queue.size() == channels) {
                vec_t v;
                for (int c=0; c<channels; c++)
                    v.insert(v.end(), queue[c].begin<float>(), queue[c].end<float>());
                queue.pop_front();
                vec_t p = nn.predict(v);
                int id=0; float mp=0;
                for (int i=0; i<p.size(); i++) {
                    if (p[i]>mp) {
                        mp = p[i];
                        id = i;
                    }
                }
                if (mp>0.99f)
                    cout << "predicted: " << id << " " << mp << " : " << names[id] << endl;
            }
        }
        imshow("TOP",frame);
        int k = waitKey(10);
        switch(k) {
            case 27: return 0;
            case ' ': track.release(); break;
        }
    }
      return 0;
}

template <class Optimizer>
int dnn_run(const string &json_model, const string &datapath, const string &pre_weigths, float learn, float decay, int batch_size, int channels, int freeze, bool test)
{
    using namespace tiny_dnn;
    using namespace tiny_dnn::activation;
    using namespace tiny_dnn::layers;
    typedef cross_entropy loss_t;

    network<sequential> nn;
    load_nn(nn, json_model, pre_weigths);

    Optimizer opt;
    opt.alpha = learn;

    for (int i = 0; i < nn.depth(); i++) {
        cout << "#layer: " << i << " " << (i<freeze ? "(frozen)" : "(active)") << "\n";
        cout << "type: "   << nn[i]->layer_type() << "\n";
        cout << "input: "  << nn[i]->in_size() << "(" << nn[i]->in_shape() << ")\n";
        cout << "output: " << nn[i]->out_size() << "(" << nn[i]->out_shape() << ")\n";
    }
    for (int i = 0; i < std::min(int(nn.depth()), freeze); i++) {
        nn[i]->set_trainable(false);
    }
    if (freeze != -1) {
        for (int i = freeze-1; i < int(nn.depth()); i++) {
            if (nn[i]->layer_type() == "fully-connected")
                nn[i]->weight_init(weight_init::xavier(.1));
        }
    }
    // load data
    vector<vec_t>   t_data,   v_data,   tt_data;
    vector<label_t> t_labels, v_labels, tt_labels;
    Mat_<int> hist_tr(1,8); hist_tr=0;
    Mat_<int> hist_tt(1,8); hist_tt=0;
    int k=0;
    FileStorage fs(datapath,0);
    FileNode n = fs["sequences"];
    FileNodeIterator si = n.begin();
    for (; si != n.end(); ++si) {
        Mat lbl;
        (*si)["labels"] >> lbl;

        FileNode p = (*si)["points"];
        FileNodeIterator pi = p.begin();
        vector<Mat> mats;
        for (; pi != p.end(); ++pi) {
            Mat m;
            (*pi) >> m;
            mats.push_back(m);
            if (m.rows==136) // i forgot to normalize the landmarks
                m /= 200;    //   based on 200x200 image
        }
        // augment data (mirror in time):
        int l=int(mats.size())-1;
        for (; l>=0; l--) {
            mats.push_back(mats[l]);
            lbl.push_back(lbl.at<int>(l));
        }

        // put "sliding windows" of the sequence into channels, [0,1,2] [1,2,3] [2,3,4], etc
        for (int l=0; l<int(mats.size())-channels; l++) {
            vec_t v;
            int id = lbl.at<int>(mats.size()/2);
            if (HAVE_NEUTRAL && (l<2 || l>int(mats.size())-channels-2)) {
                id = 0;
                if (channels > 1) {
                    v.reserve(channels*mats[0].total());
                    for (int c=0; c<channels; c++) {
                        v.insert(v.end(), mats[l].begin<float>(), mats[l].end<float>());
                    }
                } else {
                    v.insert(v.end(), mats[l].begin<float>(), mats[l].end<float>());
                }
            } else {
                if (channels > 1) {
                    v.reserve(channels*mats[0].total());
                    for (int c=0; c<channels; c++) {
                        v.insert(v.end(), mats[l+c].begin<float>(), mats[l+c].end<float>());
                    }
                } else {
                    v.insert(v.end(), mats[l].begin<float>(), mats[l].end<float>());
                }
            }

            if (k%5==0) { // 80/20 split
                v_data.push_back(v);
                v_labels.push_back(id);
                hist_tt(0,id) ++;
            } else {
                if (k%6==0) { // can't have a 'loss' for the training, so reserve some train samples for testing later
                    tt_data.push_back(v);
                    tt_labels.push_back(id);
                }
                t_data.push_back(v);
                t_labels.push_back(id);
                hist_tr(0,id) ++;
            }
            k++;
        }
    }
    fs.release();
    cout << "classes: " << hist_tr << " " << hist_tt << endl;

    if (test) {
        nn.test(v_data, v_labels).print_detail(cout);
        return 0;
    }

    timer t;
    size_t z=0; // samples per epoch
    size_t epochs = 0;
    size_t count = 0; // overall samples in this training pass
    float best_result=0;

    auto on_enumerate_epoch = [&](){
        opt.alpha *= decay;  // continuously decay learning rate
        opt.alpha = std::max((tiny_dnn::float_t)0.00001, opt.alpha);
        std::cout << "epoch " << epochs << " " << count << " samples " << t.elapsed();
        std::cout << " seconds, " << opt.alpha << " alpha, ";

        result res = nn.test(tt_data, tt_labels);
        float accuracy = (float(res.num_success) / res.num_total);
        cout << accuracy << " train acc, ";

        res = nn.test(v_data, v_labels);
        accuracy = (float(res.num_success) / res.num_total);
        cout << accuracy << " test acc" << endl;
        if (epochs%5==0)res.print_detail(cout);

        // save weights
        if (accuracy > best_result && epochs > 5) {
            std::ofstream ofs(format("weights_%d.net", epochs).c_str());
            ofs << nn;
            best_result = accuracy;
        }

        for (int i = 0; i < nn.depth(); i++) {
            if (nn[i]->layer_type() == "conv")
                nn.at<convolutional_layer>(i).weight_to_image().save(format("weights_%d.png", i).c_str());
            nn[i]->output_to_image().save(format("output_%d.png", i).c_str());
        }
        t.restart();
        z = 0; // reset local counter
        epochs ++;
    };

    auto on_enumerate_data = [&](){
        cout << (z+=batch_size) << "\r"; // local
        count += batch_size;             // global
    };

    nn.train<loss_t>(opt, t_data, t_labels, batch_size, 1000, on_enumerate_data, on_enumerate_epoch);
    return 0;
}

int main(int argc, char**argv) {
    using namespace tiny_dnn;

    CommandLineParser parser(argc, argv,
        "{ help h usage ? |      | show this message }"
        "{ video v        |      | test trained model with webcam }"
        "{ neutral n      |      | add neutral labels for real world training }"
        "{ freeze f       |    -1| freeze layers up to this one }"
        "{ test t         |      | do a test run (only) }"
        "{ channels c     |     3| batch size for dnn training }"
        "{ batch b        |    24| batch size for dnn training }"
        "{ learn l        |0.0004| initial learning rate for dnn }"
        "{ decay D        |0.95  | decay learning rate per epoch }"
        "{ weights w      |      | pretrained weights file (my.net) }"
        "{ optimizer o    |grad  | optimizer for dnn training }"
        "{ json j         |tsc32.txt| json model file for dnn (required) }"
        "{ data d         |ckplus_img.yml.gz|}");

    string json(parser.get<string>("json"));
    if (parser.has("help") || json.empty())
    {
        parser.printMessage();
        return 0;
    }
    const string datapath(parser.get<string>("data"));
    const string saved(parser.get<string>("weights"));
    const string opt(parser.get<string>("optimizer"));
    int freeze = parser.get<int>("freeze");
    int channels = parser.get<int>("channels");
    int batch_size = parser.get<int>("batch");
    float learn = parser.get<float>("learn");
    float decay = parser.get<float>("decay");
    bool test = parser.has("test");

    if (parser.has("video")) {
        return videoTest(json, saved, channels);
    }

    if (opt == "rms")
        return dnn_run<RMSprop>(json, datapath, saved, learn, decay, batch_size, channels, freeze, test);
    if (opt == "adam")
        return dnn_run<adam>(json, datapath, saved, learn, decay, batch_size, channels, freeze, test);
    if (opt == "adagrad")
        return dnn_run<adagrad>(json, datapath, saved, learn, decay, batch_size, channels, freeze, test);
    if (opt == "momentum")
        return dnn_run<momentum>(json, datapath, saved, learn, decay, batch_size, channels, freeze, test);
    return dnn_run<gradient_descent>(json, datapath, saved, learn, decay, batch_size, channels, freeze, test);
}


// epoch 139 893760 samples 54.5469 seconds, 1e-005 alpha, 0.909864 accuracy.


/*
ckplus_tiny -l=0.0001 -w=weights_29.net -c=3 -j=ck3.txt -f=13 -o=adam -D=0.99 -t

classes: [0, 709, 142, 556, 379, 898, 370, 862] [0, 178, 37, 135, 92, 226, 93, 218]
accuracy:99.285% (972/979)
    *     1     2     3     4     5     6     7
    1   177     0     0     0     0     0     0
    2     0    37     0     0     0     0     0
    3     0     0   134     0     0     0     2
    4     0     0     1    92     0     0     1
    5     0     0     0     0   225     0     0
    6     0     0     0     0     0    93     1
    7     1     0     0     0     1     0   214
*/
