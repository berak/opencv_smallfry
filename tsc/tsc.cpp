//
// belgium traffic sign classification set from:
//   http://btsd.ethz.ch/shareddata/
//   ( BelgiumTSC_Training.zip & BelgiumTSC_Testing.zip
//     i changed the directory structure, so it looks like:
//     BelgiumTSC
//     ├───Testing
//     │   ├───00000
//     │   ├───00001
//     ...
//     │   └───00061
//     └───Training
//         ├───00000
//         ├───00001
//     ...
//         └───00061 )
// dnn code from:
//   https://github.com/tiny-dnn/
//   (main tries to load a json model)
//


#include "tiny_dnn/tiny_dnn.h" // order matters, this has to go before opencv (ACCESS_WRITE)
#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>

using namespace cv;
using namespace std;
using tiny_dnn::vec_t;
using tiny_dnn::label_t;

int WINSIZE=32;
String tscdir = "C:/data/BelgiumTSC/";

//! convert a single image plane to vec_t for tiny_dnn
void convert_plane(const Mat &img, vec_t &d)
{
    Mat_<uchar> image = img.reshape(1,1);
    std::transform(image.begin(), image.end(), std::back_inserter(d),
                   [=](uint8_t c) { return float(c)/255.0f; });
}
//! convert (possibly color) image to vec_t for tiny_dnn, and add to dataset
void convert_image(const Mat &img, int lab, std::vector<vec_t>& data, std::vector<label_t>& labels)
{
    vec_t d;
    if (img.channels() == 1) {
        convert_plane(img, d);
    } else {
        vector<Mat> chn; split(img, chn);
        for (auto c : chn) {
            convert_plane(c, d);
        }
    }
    data.push_back(d);
    labels.push_back(lab);
    cout << "ann " << data.size() << "\t" << lab << "\r";
}
//! convert image to Mat for cv::ml::SVM, and add to dataset
void convert_image(const Mat &image, int lab, cv::Mat &data, cv::Mat &labels)
{
    data.push_back(image.reshape(1,1));
    labels.push_back(lab);
    cout << "svm " << data.rows << "\t" << lab << "\r";
}

//
//! load images for maxn classes from train or test dir.
// note, that the csv files *claim* to have W,h,x,y order, but it's the other way round ! (H,W,y,x)
//
template<class Datatype, class Labelstype>
double load(const String &dir, Datatype &data, Labelstype &labels, int maxn=-1, bool gray=true)
{
    int64 t = getTickCount();

    vector<String> csvs;
    glob(dir + "*.csv",csvs, true);
    for (auto cn : csvs) {
        ifstream csv(cn);
        string file;
        getline(csv,file);
        while(csv.good()) {
            char c = 0;
            file = "";
            while (csv.good()) {
                csv.get(c);
                if (c == ';') break;
                if (c == '\n') continue;
                file += c;
            }
            if (file.empty()) break;
            int W,H,x1,y1,x2,y2,l;
            csv >> H >> c >> W >> c >> y1 >> c >> x1 >> c >> y2 >> c >> x2 >> c >> l;
            if (maxn>0 && l>=maxn) break;
            Rect r(Point(x1,y1), Point(x2,y2));
            String fn = dir + format("%05d/",l) + file;
            Mat img = imread(fn, (gray?0:1));
            if (img.empty()) {
                //cout << "no " << fn << endl;
                continue;
            }
            cv::Mat resized;
            cv::resize(img(r), resized, cv::Size(WINSIZE, WINSIZE));
            convert_image(resized,l,data,labels);
        }
    }
    cout << endl;

    int64 t1 = getTickCount();
    return  ((t1-t)/getTickFrequency());
}


//
//! load a json model from file, reconfigure traindata settings (winsize, maxn)
//  optionally load pretrained weights
//
int ann_test(int maxn, char *json, char *savedata, float learn)
{
    using namespace tiny_dnn;
    using namespace tiny_dnn::activation;
    using namespace tiny_dnn::layers;

    network<sequential> nn;
    try {
        nn.load(json, content_type::model, file_format::json);
        std::vector<shape3d> shp_in = nn[0]->in_data_shape();
        WINSIZE = shp_in[0].width_;
        int last = int(nn.layer_size()) - 1;
        std::vector<shape3d> shp_out = nn[last]->out_data_shape();
        maxn = shp_out[0].width_;
        cout << "in " << WINSIZE << ", out " << maxn << endl;
        if (savedata) {
            ifstream ifs(savedata);
            ifs >> nn;
        } else {
            nn.weight_init(weight_init::xavier(0.2));
        }
        nn.save("mymodel.txt", content_type::model, file_format::json);
    } catch (const nn_error& e) {
       std::cout << e.what();
    }

    /// i wich i could configure this dynamically !
    //gradient_descent optimizer;
    //momentum optimizer;
    adagrad opt;
    opt.alpha = learn;
    //optimizer.lambda = 0.05;
    //optimizer.mu = 0.85;

    for (int i = 0; i < nn.depth(); i++) {
        cout << "#layer:" << i << "\n";
        cout << "layer type:" << nn[i]->layer_type() << "\n";
        cout << "input:" << nn[i]->in_size() << "(" << nn[i]->in_shape() << ")\n";
        cout << "output:" << nn[i]->out_size() << "(" << nn[i]->out_shape() << ")\n";
    }

    // load data
    vector<vec_t> data, v_data;
    vector<label_t> labels, v_labels;

    double tl = load(tscdir + "Training/",data,labels,maxn, false);
    int n = data.size() * data[0].size();
    cout << data.size() << " train elems, " << n << " bytes. " << tl << " seconds. " << endl;

    tl = load(tscdir + "Testing/",v_data,v_labels,maxn, false);
    n = v_data.size() * v_data[0].size();
    cout << v_data.size() << " test  elems, " << n << " bytes. " << tl << " seconds." <<endl;

    timer t;
    size_t z=0; // samples seen per epoch
    size_t batch_size = 12;
    size_t epochs = 0;
    size_t count = 0; // overall samples seen in this training pass

    // test accuracy on (a few) random samples
    auto check = [&](const string &tit, const vector<vec_t> &data, const vector<label_t> &labels){
        int ntests = 100;
        int correct = 0;
        for (int i=0; i<ntests; i++) {
            size_t n = size_t(rand()) % data.size();
            try {
                size_t pid = nn.predict_label(data[n]);
                size_t tid = labels[n];
                correct += (pid == tid);
            } catch (const nn_error& e) {
                std::cout << i << " " << n << " " << data[n].size() << "\n" << e.what();
            }
        }
        float acc = float(correct) / float(ntests);
        cout << tit << " " << acc << " " ;
    };

    auto on_enumerate_epoch = [&](){
        opt.alpha *= 0.98;  // decay learning rate
        opt.alpha = std::max((tiny_dnn::float_t)0.00001, opt.alpha);
        std::cout << "epoch " << epochs << " " << count << " samples " << t.elapsed();
        std::cout << " seconds, " << opt.alpha << " alpha. ";
        epochs ++;

        check("train", data,labels);
        check("valid", v_data,v_labels);
        result res = nn.test(v_data, v_labels);
        cout << "test " << (float(res.num_success)/res.num_total) << endl;

        nn[0]->output_to_image().write("layer0.bmp");
        nn[1]->output_to_image().write("layer1.bmp");
        nn[2]->output_to_image().write("layer2.bmp");
        nn[3]->output_to_image().write("layer3.bmp");

/* WIP!!
        std::vector<vec_t*> w0 = nn[0]->weights();
        Mat draw;
        cout << w0.size() << " elems. " << w[0]->size() << "features.";
        for (auto f=w0.begin(); f!=w0.end(); f++) {
            const vec_t *v = (*f);
            int w = (int)sqrt(double(v->size() + 4));
            Mat m = Mat(w, w, CV_32F, (void*)v->data());
            if (draw.empty()) draw = m.clone();
            else hconcat(draw, m, draw);
        }
        imshow("filters 0", draw);
        waitKey(5);
*/
        // save weights
        std::ofstream ofs("my.net");
        ofs << nn;

        t.restart();
        z = 0;
    };

    auto on_enumerate_data = [&](){
        cout << (z+=batch_size) << "\r";
        count += batch_size;
    };

    nn.train<cross_entropy>(opt, data, labels, batch_size, 1000,
                  on_enumerate_data, on_enumerate_epoch);

    return 0;
}


//
// cross-checking with opencv's SVM (0.93 with first 20 classes only)
//
int svm_test(int maxn)
{
    Mat data,labels;
    double t = load(tscdir + "Training/",data,labels,maxn);
    data.convertTo(data, CV_32F);
    int n = data.total() * data.elemSize();
    cout << data.rows << " elems, " << n << " bytes." << endl;

    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setKernel(ml::SVM::LINEAR);
    svm->train(data,0,labels);

    data.release();
    labels.release();
    t = load(tscdir + "Testing/",data,labels,maxn);
    data.convertTo(data, CV_32F);
    n = data.total() * data.elemSize();
    cout << data.rows << " elems, " << n << " bytes." << endl;

    Mat results;
    svm->predict(data,results);
    results.convertTo(results, CV_32S);
    float correct = countNonZero(results == labels);
    float accuracy = correct / results.total();
    cout << "accuracy: " << accuracy << endl;
    int C = 62;
    if (maxn!=-1) C=maxn;
    Mat_<int> confusion(C,C,0);
    for (int i=0; i<results.rows; i++) {
        int p = (int)results.at<int>(i);
        int t = (int)labels.at<int>(i);
        confusion(p,t) ++;
    }
    cout << "confusion:\n" << confusion << endl;
    return 0;
}


int main(int argc, char **argv)
{
    int maxn=20;
    char *json = (char*)"mymodel.txt";
    if (argc>1) json = argv[1];
    float learn = 0.01f;
    if (argc>2) learn = atof(argv[2]);
    char *save = 0;
    if (argc>3) save = argv[3];
    return ann_test(maxn, json, save, learn);
    //return svm_test(maxn);
}
