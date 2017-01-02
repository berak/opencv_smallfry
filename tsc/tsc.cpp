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
// also comes with opencv SVM & ANN_MLP reference impl.
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
    const float scale = 1.0f / 255.0f;
    Mat_<uchar> image = img.reshape(1, 1);
    std::transform(image.begin(), image.end(), std::back_inserter(tiny_vec),
                   [=](uint8_t c) { return float(c) * scale; });
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
void add_image(const Mat &image, int lab, std::vector<vec_t>& data, std::vector<label_t>& labels)
{
    vec_t tiny_vec;
    convert_image(image, tiny_vec);
    data.push_back(tiny_vec);
    labels.push_back(lab);
    cout << "dnn " << data.size() << "\t" << lab << "\r";
}


//! convert image Mat for cv::ml classes, and add to dataset
void add_image(const Mat &image, int lab, cv::Mat &data, cv::Mat &labels)
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
double load(const String &dir, Datatype &data, Labelstype &labels, int max_classes=-1, bool gray=true, int skip=0)
{
    int64 t0 = getTickCount();
    int k = 0;
    vector<String> csvs;
    glob(dir + "*.csv", csvs, true);
    for (auto cn : csvs) {
        ifstream csv(cn);
        string file;
        getline(csv, file); // skip csv header
        while(csv.good()) {
            k++;
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

            int W,H,x1,y1,x2,y2,label;
            csv >> H >> c >> W >> c >> y1 >> c >> x1 >> c >> y2 >> c >> x2 >> c >> label;
            if ((skip > 1) && (k % skip != 0)) continue;
            if ((max_classes > 0) && (label >= max_classes)) break;

            String fn = dir + format("%05d/",label) + file;
            Mat img = imread(fn, (gray?0:1));
            if (img.empty()) continue;

            cv::Mat resized;
            Rect roi(Point(x1,y1), Point(x2,y2));
            cv::resize(img(roi), resized, cv::Size(WINSIZE, WINSIZE));
            add_image(resized, label, data, labels);
        }
    }
    int64 t1 = getTickCount();
    return  ((t1-t0)/getTickFrequency());
}


//! load a json model from file, adjust traindata settings (winsize, max_classes)
//!  optionally load pretrained weights
int dnn(int max_classes, char *json_model, char *pre_weigths, float learn, const string &op)
{
    using namespace tiny_dnn;
    using namespace tiny_dnn::activation;
    using namespace tiny_dnn::layers;
    typedef cross_entropy loss_t;

    network<sequential> nn;
    try {
        nn.load(json_model, content_type::model, file_format::json);
        std::vector<shape3d> shp_in = nn[0]->in_data_shape();
        WINSIZE = shp_in[0].width_;
        int last = int(nn.layer_size()) - 1;
        std::vector<shape3d> shp_out = nn[last]->out_data_shape();
        max_classes = shp_out[0].width_;

        if (pre_weigths) {
            ifstream ifs(pre_weigths);
            ifs >> nn;
        } else {
            nn.weight_init(weight_init::xavier(1));
            //nn.weight_init(weight_init::lecun());
        }

        nn.save("mymodel.txt", content_type::model, file_format::json);
    } catch (const nn_error& e) {
       std::cout << e.what();
    }

    // yea, this is horrible, but allows me to specify an optimizer from cmdline :[
    map<string,optimizer*> ops;
    ops["rms"]      = new RMSprop();
    ops["grad"]     = new gradient_descent();
    ops["adam"]     = new adam();
    ops["adagrad"]  = new adagrad();
    ops["momentum"] = new momentum();
    auto get_alpha = [&](const string &op) {
        if (op=="rms") return &(static_cast<RMSprop*>(ops[op])->alpha);
        if (op=="grad") return &(static_cast<gradient_descent*>(ops[op])->alpha);
        if (op=="adam") return &(static_cast<adam*>(ops[op])->alpha);
        if (op=="adagrad") return &(static_cast<adagrad*>(ops[op])->alpha);
        if (op=="momentum") return &(static_cast<momentum*>(ops[op])->alpha);
        static float _ = 0.0f;
        return &_;
    };
    *get_alpha(op) = learn;

    cout << op << ", in " << WINSIZE << ", out " << max_classes << endl;
    for (int i = 0; i < nn.depth(); i++) {
        cout << "#layer: " << i << "\n";
        cout << "type: "   << nn[i]->layer_type() << "\n";
        cout << "input: "  << nn[i]->in_size() << "(" << nn[i]->in_shape() << ")\n";
        cout << "output: " << nn[i]->out_size() << "(" << nn[i]->out_shape() << ")\n";
    }

    // load data
    vector<vec_t>   t_data,   v_data;
    vector<label_t> t_labels, v_labels;

    double tl = load(tscdir + "Training/", t_data, t_labels, max_classes, false);
    int n = t_data.size() * t_data[0].size();
    cout << "dnn train " << t_data.size() << " samples, " << n << " bytes. " << tl << " seconds. " << endl;

    tl = load(tscdir + "Testing/", v_data, v_labels, max_classes, false, 3);
    n = v_data.size() * v_data[0].size();
    cout << "dnn test  " << v_data.size() << " samples, " << n << " bytes. " << tl << " seconds." <<endl;

    timer t;
    size_t z=0; // samples per epoch
    size_t batch_size = 24;
    size_t epochs = 0;
    size_t count = 0; // overall samples in this training pass

    // test accuracy on (a few) random samples
    auto check = [&](const string &title, const vector<vec_t> &data, const vector<label_t> &labels){
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
        cout << title << " " << acc << " " ;
    };

    auto on_enumerate_epoch = [&](){
        float *alpha = get_alpha(op);
        *alpha *= 0.95;  // continuously decay learning rate
        *alpha = std::max((tiny_dnn::float_t)0.00001, *alpha);
        std::cout << "epoch " << epochs << " " << count << " samples " << t.elapsed();
        std::cout << " seconds, " << *alpha << " alpha. ";
        epochs ++;

        check("train", t_data, t_labels);
        //check("valid", v_data, v_labels);
        result res = nn.test(v_data, v_labels);
        cout << "test " << (float(res.num_success) / res.num_total) << endl;
        //double loss = nn.get_loss<loss_t>(v_data, v_labels);
        //cout << "loss " << loss << endl;

        nn[0]->output_to_image().write("layer0.bmp");
        nn[1]->output_to_image().write("layer1.bmp");
        nn[2]->output_to_image().write("layer2.bmp");
        nn[3]->output_to_image().write("layer3.bmp");
        try {
            nn.at<conv<leaky_relu>>(0).weight_to_image().write("layer0_w.bmp");
            nn.at<conv<leaky_relu>>(1).weight_to_image().write("layer1_w.bmp");
        } catch (const nn_error& e) {
           cout << e.what();
        }

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

    nn.train<loss_t>(*ops[op], t_data, t_labels, batch_size, 1000,
                  on_enumerate_data, on_enumerate_epoch);

    return 0;
}


//! process opencv prediction results:
void cv_results(int max_classes, Mat &results, Mat &labels, const String &title)
{
    // confusion:
    int C = std::min(max_classes, 62);
    Mat_<int> confusion(C, C, 0);
    for (int i=0; i<results.rows; i++) {
        int p = (int)results.at<float>(i);
        int t = (int)labels.at<int>(i);
        confusion(p, t) ++;
    }
    const int MAXC = 30; // try to fit console win
    if (confusion.rows < MAXC)
        cout << title << "confusion:\n" << confusion << endl;
    else // skip results beyond MAXC
        cout << title << format("confusion (%d cropped to %d:\n", confusion.rows, MAXC) << confusion(Rect(0, 0, MAXC, MAXC)) << endl;

    // accuracy:
    float correct  = sum(confusion.diag())[0];
    float accuracy = correct / results.rows;
    cout << title << "accuracy: " << accuracy << endl;
}


//! load & print stats
void cv_load(const String &dir, Mat &data, Mat &labels, int max_classes, const String &title)
{
    data.release();
    labels.release();
    double t = load(tscdir + dir, data, labels, max_classes);
    int n = data.total() * data.elemSize();
    cout << title << data.rows << " elems, " << n << " bytes, " << max_classes << " classes, " << t <<  " seconds." << endl;
}

using tiny_dnn::timer;

int cv_svm(int max_classes)
{
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setKernel(ml::SVM::LINEAR);

    Mat data, labels;
    cv_load("Training/", data, labels, max_classes, "svm train ");

    timer t;
    svm->train(data, 0, labels);
    double t1 = t.elapsed();

    cv_load("Testing/", data, labels, max_classes, "svm test  ");

    t.restart();
    Mat results;
    svm->predict(data, results);
    double t2 = t.elapsed();

    cout << "svm " << t1 << " / " << t2 << " seconds." << endl;
    cv_results(max_classes, results, labels, "svm ");
    return 0;
}


int cv_mlp(int max_classes)
{
    Mat_<int> layers(3, 1);
    layers << WINSIZE*WINSIZE, 200, max_classes;

    Ptr<ml::ANN_MLP> nn = ml::ANN_MLP::create();
    nn->setLayerSizes(layers);
    nn->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);
    nn->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM);
    nn->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 300, 0.0001));

    Mat data, labels;
    cv_load("Training/", data, labels, max_classes, "mlp train ");

    // mlp needs "one-hot" encoded responses for training
    Mat hot(labels.rows, max_classes, CV_32F, 0.0f);
    for (int i=0; i<labels.rows; i++)
    {
        int id = (int)labels.at<int>(i);
        hot.at<float>(i, id) = 1.0f;
    }
    timer t;
    nn->train(data, 0, hot);
    double t1 = t.elapsed();

    cv_load("Testing/", data, labels, max_classes, "mlp test  ");
    t.restart();
    Mat results;
    // doing single predictions is slower, but this avoids having to unroll the result
    for (int r=0; r<data.rows; r++) {
        float p = nn->predict(data.row(r));
        results.push_back(p);
    }
    double t2 = t.elapsed();
    cout << "mlp " << t1 << " / " << t2 << " seconds." << endl;

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

    char *json = (char*)"tsc32.txt";
    if (argc>1) json = argv[1];
    string opt("grad");
    if (argc>2) opt = argv[2];
    float learn = 0.01f;
    if (argc>3) learn = atof(argv[3]);
    char *saved = 0;
    if (argc>4) saved = argv[4];

    return dnn(max_classes, json, saved, learn, opt);
}
