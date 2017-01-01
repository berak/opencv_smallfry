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

using namespace cv;
using namespace std;
using tiny_dnn::vec_t;
using tiny_dnn::label_t;

int WINSIZE   = 24; // dnn/json will override this
String tscdir = "C:/data/BelgiumTSC/";

//! convert a single cv::Mat image plane to tiny_dnn::vec_t
void convert_plane(const Mat &img, vec_t& tiny_vec)
{
    Mat_<uchar> image = img.reshape(1,1);
    std::transform(image.begin(), image.end(), std::back_inserter(tiny_vec),
                   [=](uint8_t c) { return float(c)/255.0f; });
}

//! convert (possibly color) image to tiny_dnn::vec_t, with consecutive image planes
void convert_image(const Mat &image, vec_t & tiny_vec)
{
    if (image.channels() == 1) {
        convert_plane(image, tiny_vec);
    } else {
        vector<Mat> chn;
        split(image, chn);
        for (auto c : chn) {
            convert_plane(c, tiny_vec);
        }
    }
}

//! convert Mat to vec_t for tiny_dnn, and add to dataset
//  tiny_dnn wants consecutive image planes
void convert_image(const Mat &image, int lab, std::vector<vec_t>& data, std::vector<label_t>& labels)
{
    vec_t tiny_vec;
    convert_image(image, tiny_vec);
    data.push_back(tiny_vec);
    labels.push_back(lab);
    cout << "dnn " << data.size() << "\t" << lab << "\r";
}

//! convert image Mat for cv::ml classes, and add to dataset
void convert_image(const Mat &image, int lab, cv::Mat &data, cv::Mat &labels)
{
    Mat img;
    image.convertTo(img, CV_32F, 1.0/255);
    data.push_back(img.reshape(1,1));
    labels.push_back(lab);
    cout << "mat " << data.rows << "\t" << lab << "\r";
}


//! load images for max_classes classes from train or test dir.
//  note, that the csv files *claim* to have W,h,x,y order, but actually, it's the other way round ! (H,W,y,x)
template<class Datatype, class Labelstype>
double load(const String &dir, Datatype &data, Labelstype &labels, int max_classes=-1, bool gray=true)
{
    int64 t = getTickCount();

    vector<String> csvs;
    glob(dir + "*.csv", csvs, true);
    for (auto cn : csvs) {
        ifstream csv(cn);
        string file;
        getline(csv, file); // skip csv header
        while(csv.good()) {
            char c = 0;
            file = "";
            // avoid "evil string globbing"
            while (csv.good()) {
                csv.get(c);
                if (c == ';') break;
                if (c == '\n') continue;
                file += c;
            }
            if (file.empty()) break;

            int W,H,x1,y1,x2,y2,l;
            csv >> H >> c >> W >> c >> y1 >> c >> x1 >> c >> y2 >> c >> x2 >> c >> l;
            if (max_classes>0 && l>=max_classes) break;
            Rect r(Point(x1,y1), Point(x2,y2));

            String fn = dir + format("%05d/",l) + file;
            Mat img = imread(fn, (gray?0:1));
            if (img.empty()) continue;

            cv::Mat resized;
            cv::resize(img(r), resized, cv::Size(WINSIZE, WINSIZE));
            convert_image(resized, l, data, labels);
        }
    }
//    cout << endl;

    int64 t1 = getTickCount();
    return  ((t1-t)/getTickFrequency());
}


//! load a json model from file, adjust traindata settings (winsize, max_classes)
//!  optionally load pretrained weights
int dnn(int max_classes, char *json, char *pre_weigths, float learn)
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
        max_classes = shp_out[0].width_;
        cout << "in " << WINSIZE << ", out " << max_classes << endl;

        if (pre_weigths) {
            ifstream ifs(pre_weigths);
            ifs >> nn;
        } else {
            nn.weight_init(weight_init::xavier(0.2));
        }

        nn.save("mymodel.txt", content_type::model, file_format::json);
    } catch (const nn_error& e) {
       std::cout << e.what();
    }

    /// WIP - i wish i could configure this from json !
    //gradient_descent optimizer;
    //momentum optimizer;
    adagrad opt;
    opt.alpha = learn;
    //optimizer.lambda = 0.05;
    //optimizer.mu = 0.85;

    for (int i = 0; i < nn.depth(); i++) {
        cout << "#layer: " << i << "\n";
        cout << "type: "   << nn[i]->layer_type() << "\n";
        cout << "input: "  << nn[i]->in_size() << "(" << nn[i]->in_shape() << ")\n";
        cout << "output: " << nn[i]->out_size() << "(" << nn[i]->out_shape() << ")\n";
    }

    // load data
    vector<vec_t> data, v_data;
    vector<label_t> labels, v_labels;

    double tl = load(tscdir + "Training/", data, labels, max_classes, false);
    int n = data.size() * data[0].size();
    cout << "dnn train " << data.size() << " elems, " << n << " bytes. " << tl << " seconds. " << endl;

    tl = load(tscdir + "Testing/", v_data, v_labels, max_classes, false);
    n = v_data.size() * v_data[0].size();
    cout << "dnn test  " << v_data.size() << " elems, " << n << " bytes. " << tl << " seconds." <<endl;

    timer t;
    size_t z=0; // samples per epoch
    size_t batch_size = 12;
    size_t epochs = 0;
    size_t count = 0; // overall samples in this training pass

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

        check("train", data, labels);
        check("valid", v_data, v_labels);
        result res = nn.test(v_data, v_labels);
        cout << "test " << (float(res.num_success) / res.num_total) << endl;

        nn[0]->output_to_image().write("layer0_w.bmp");
        nn[1]->output_to_image().write("layer1.bmp");
        nn[2]->output_to_image().write("layer2.bmp");
        nn[3]->output_to_image().write("layer3.bmp");
        /// WIP
        //nn.at<conv<leaky_relu>(0)->weight_to_image().write("layer0_w.bmp");
        //nn.at<conv<leaky_relu>(1)->weight_to_image().write("layer1_w.bmp");

        /** WIP
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
        **/

        // save weights
        std::ofstream ofs("my.net");
        ofs << nn;

        t.restart();
        z = 0; // reset local counter
    };

    auto on_enumerate_data = [&](){
        cout << (z+=batch_size) << "\r"; // local
        count += batch_size;             // global
    };

    nn.train<cross_entropy>(opt, data, labels, batch_size, 1000,
                  on_enumerate_data, on_enumerate_epoch);

    return 0;
}


//! process opencv prediction results:
void cv_results(int max_classes, Mat &results, Mat &labels, const String &tit)
{
    // confusion:
    int C = std::min(max_classes, 62);
    Mat_<int> confusion(C, C, 0);
    for (int i=0; i<results.rows; i++) {
        int p = (int)results.at<float>(i);
        int t = (int)labels.at<int>(i);
        confusion(p, t) ++;
    }
    const int MAXC = 30;
    if (confusion.rows < MAXC)
        cout << tit << "confusion:\n" << confusion << endl;
    else // skip results beyond MAXC
        cout << tit << "confusion:\n" << confusion(Rect(0, 0, MAXC, MAXC)) << endl;

    // accuracy:
    float correct  = sum(confusion.diag())[0];
    float accuracy = correct / results.rows;
    cout << tit << "accuracy: " << accuracy << endl;
}


//! load & print stats
void cv_load(const String &dir, Mat &data, Mat &labels, int max_classes, const String &tit)
{    data.release();
    labels.release();
    double t = load(tscdir + dir, data, labels, max_classes);
    int n = data.total() * data.elemSize();
    cout << tit << data.rows << " elems, " << n << " bytes, " << max_classes << " classes, " << t <<  " seconds." << endl;
}

int cv_svm(int max_classes)
{
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setKernel(ml::SVM::LINEAR);

    Mat data, labels;
    cv_load("Training/", data, labels, max_classes, "svm train ");
    svm->train(data, 0, labels);

    cv_load("Testing/", data, labels, max_classes, "svm test  ");
    Mat results;
    svm->predict(data, results);
    cv_results(max_classes, results, labels, "svm ");
    return 0;
}

int cv_mlp(int max_classes)
{
    Ptr<ml::ANN_MLP> nn = ml::ANN_MLP::create();
    Mat_<int> layers(3, 1);
    layers << WINSIZE*WINSIZE, 400, max_classes;
    nn->setLayerSizes(layers);
    nn->setTrainMethod(ml::ANN_MLP::BACKPROP);
    nn->setTermCriteria(TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 1000, 0.00001f));
    nn->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 1, 1);
    nn->setBackpropWeightScale(0.1f);
    nn->setBackpropMomentumScale(0.1f);

    Mat data, labels;
    cv_load("Training/", data, labels, max_classes, "mlp train ");

    // mlp needs one-hot encoded responses
    Mat hot(labels.rows, max_classes, CV_32F, 0.0f); // all zero, initially
    for (int i=0; i<labels.rows; i++)
    {
        int id = (int)labels.at<int>(i);
        hot.at<float>(i, id) = 1.0f;
    }
    nn->train(data, 0, hot);

    cv_load("Testing/", data, labels, max_classes, "mlp test  ");
    Mat results;
    nn->predict(data, results);
    cv_results(max_classes, results, labels, "mlp ");
    return 0;
}


int main(int argc, char **argv)
{
    int max_classes = 20; // restrict to first n classes

    if (argc>1 && argv[1][0] == 's')
        return cv_svm(max_classes);
    if (argc>1 && argv[1][0] == 'm')
        return cv_mlp(max_classes);

    char *json = (char*)"mymodel.txt";
    if (argc>1) json = argv[1];
    float learn = 0.01f;
    if (argc>2) learn = atof(argv[2]);
    char *save = 0;
    if (argc>3) save = argv[3];

    dnn(max_classes, json, save, learn);
    return 0;
}
