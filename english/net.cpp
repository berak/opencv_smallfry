#include "tiny_dnn/tiny_dnn.h" // order matters, on win, this has to go before opencv (ACCESS_WRITE)
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using tiny_dnn::vec_t;
using tiny_dnn::label_t;


int WINSIZE   = 32; // dnn/json will override this
String english = "C:/data/English/Img/GoodImg/Bmp/";


//! convert a single cv::Mat image plane to tiny_dnn::vec_t
void convert_plane(const Mat &img, vec_t& tiny_vec)
{
    const float scale = 1.0f / 255.0f;
    Mat_<uchar> image = img.reshape(1, 1);
    std::transform(image.begin(), image.end(), std::back_inserter(tiny_vec),
                   [=](uint8_t c) { return float(c) * scale; });
}


//! convert (possibly color) image to tiny_dnn::vec_t, with consecutive image planes
void convert_image(const Mat &image, vec_t &vec)
{
    if (image.channels() == 1) {
        convert_plane(image, vec);
    } else {
        vector<Mat> chn;
        split(image, chn);
        for (auto c : chn) {
            convert_plane(c, vec);
        }
    }
}


//! convert Mat to vec_t for tiny_dnn, and add to dataset
//  tiny_dnn wants consecutive image planes
void add_image(const Mat &image, int lab, std::vector<vec_t>& data, std::vector<label_t>& labels)
{
    vec_t vec;
    convert_image(image, vec);
    data.push_back(vec);
    labels.push_back(lab);
    cout << "dnn " << data.size() << "\t" << lab << "\r";
}

//! load images for max_classes classes from train or validation dir.
double load(const String &path,
            std::vector<vec_t>& data, std::vector<label_t>& labels,
            std::vector<vec_t>& v_data, std::vector<label_t>& v_labels, bool gray=true, int test_ratio=6)
{
    int64 t0 = getTickCount();
    for (label_t id=0; id<62; id++) {
        String dir = path + format("Sample%03d/*.png", id+1);
        vector<String> files;
        glob(dir, files);
        int k=0;
        for (auto fn : files) {
            Mat img = imread(fn, (gray?0:1));
            if (img.empty()) continue;
            Mat resized;
            cv::resize(img, resized, cv::Size(WINSIZE, WINSIZE));
            if ((test_ratio > 0) && (k % test_ratio == 0)) {
                add_image(resized, id, v_data, v_labels);
            } else {
                add_image(resized, id, data, labels);
            }
            k ++;
        }
    }
    int64 t1 = getTickCount();
    return  ((t1-t0)/getTickFrequency());
}


//! load a json model from file, adjust traindata settings (winsize, max_classes)
//!  optionally load pretrained weights
template <class Optimizer>
int dnn_train(const string &json_model, const string &pre_weigths, float learn, float decay, int batch_size)
{
    using namespace tiny_dnn;
    using namespace tiny_dnn::activation;
    using namespace tiny_dnn::layers;
    typedef cross_entropy loss_t;

    network<sequential> nn;

    int max_classes = 74;
    try {
        nn.load(json_model.c_str(), content_type::model, file_format::json);
        std::vector<shape3d> shp_in = nn[0]->in_data_shape();
        WINSIZE = shp_in[0].width_;
        int last = int(nn.layer_size()) - 1;
        std::vector<shape3d> shp_out = nn[last]->out_data_shape();
        max_classes = shp_out[0].width_;

        if (! pre_weigths.empty()) {
            cout << "reading weights from " << pre_weigths << endl;
            ifstream ifs(pre_weigths.c_str());
            ifs >> nn;
        } else {
            //nn.weight_init(weight_init::xavier(1));
            //nn.weight_init(weight_init::lecun());
        }

        nn.save("mymodel.txt", content_type::model, file_format::json);
    } catch (const nn_error& e) {
       std::cout << e.what();
    }

    Optimizer opt;
    opt.alpha = learn;

    cout << "in " << WINSIZE << ", out " << max_classes << ", batch " << batch_size << endl;
    for (int i = 0; i < nn.depth(); i++) {
        cout << "#layer: " << i << "\n";
        cout << "type: "   << nn[i]->layer_type() << "\n";
        cout << "input: "  << nn[i]->in_size() << "(" << nn[i]->in_shape() << ")\n";
        cout << "output: " << nn[i]->out_size() << "(" << nn[i]->out_shape() << ")\n";
    }

    // load data
    vector<vec_t>   t_data,   v_data;
    vector<label_t> t_labels, v_labels;

    double tl = load(english, t_data, t_labels, v_data, v_labels, true, 6);
    cout << "dnn train data " << t_data.size() << " samples,  test data " << v_data.size() << " samples, " << tl << " seconds. " << endl;

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

        result res = nn.test(v_data, v_labels);
        float accuracy = (float(res.num_success) / res.num_total);
        cout << accuracy << " accuracy." << endl;
        //double loss = nn.get_loss<loss_t>(v_data, v_labels);
        //cout << "loss " << loss << endl;

        // save weights
        if (accuracy > best_result) {
            std::ofstream ofs("my.net");
            ofs << nn;
            best_result = accuracy;
        }

        // FIXME, this does no more exist !
        //for (int i=0; i<nn.depth()-1; i++)
        //    nn[i]->output_to_image().write(format("layer%i.bmp", i));

        t.restart();
        z = 0; // reset local counter
        epochs ++;
    };

    auto on_enumerate_data = [&](){
        cout << (z+=batch_size) << "\r"; // local
        count += batch_size;             // global
    };

    nn.train<loss_t>(opt, t_data, t_labels, batch_size, 1000,
                  on_enumerate_data, on_enumerate_epoch);
    return 0;
}


//! load a json model and pretrained weights from file, adjust traindata settings (winsize, max_classes)
//!  and predict on test images
int dnn_test(const string &json_model, const string &pre_weigths)
{
    using namespace tiny_dnn;
/*
    network<sequential> nn;
    int max_classes = 62;
    try {
        nn.load(json_model, content_type::model, file_format::json);
        std::vector<shape3d> shp_in = nn[0]->in_data_shape();
        WINSIZE = shp_in[0].width_;
        int last = int(nn.layer_size()) - 1;
        std::vector<shape3d> shp_out = nn[last]->out_data_shape();
        max_classes = shp_out[0].width_;

        cout << "reading weights from " << pre_weigths << endl;
        ifstream ifs(pre_weigths);
        ifs >> nn;
    } catch (const nn_error& e) {
       std::cout << e.what();
    }

    // load data
    vector<vec_t>   v_data;
    vector<label_t> v_labels;

    double tl = load(english + "Testing/", v_data, v_labels, max_classes, false, 1);
    int n = v_data.size() * v_data[0].size();
    cout << "dnn test  " << v_data.size() << " samples, " << n << " bytes. " << tl << " seconds." <<endl;

    timer t;
    result res = nn.test(v_data, v_labels);
    float accuracy = (float(res.num_success) / res.num_total);
    cout << "test " << accuracy << " accuracy, " << t.elapsed() << " seconds." << endl;
*/
    return 0;
}


int main(int argc, char **argv)
{
    using namespace tiny_dnn;

    CommandLineParser parser(argc, argv,
        "{ help h usage ? |      | show this message }"
        "{ test t         |      | test dnn on pretrained model }"
        "{ batch b        |    24| batch size for dnn training }"
        "{ maxc M         |    62| restrict to M classes (for speedup), only for svm,mlp }"
        "{ learn l        |0.0004| initial learning rate for dnn }"
        "{ decay d        |0.95  | decay learning rate per epoch }"
        "{ weights w      |      | pretrained weights file (my.net) }"
        "{ optimizer o    |grad  | optimizer for dnn training }"
        "{ json j         |tsc32.txt| json model file for dnn (required) }"
        "{ data D         |C:/data/English/Img/GoodImg/Bmp/| path to dataset }" );

    string json(parser.get<string>("json"));
    if (parser.has("help") || json.empty())
    {
        parser.printMessage();
        return 0;
    }
    english = parser.get<string>("data"); // global
    string saved(parser.get<string>("weights"));
    string opt(parser.get<string>("optimizer"));
    int max_classes = parser.get<int>("maxc");
    int batch_size = parser.get<int>("batch");
    float learn = parser.get<float>("learn");
    float decay = parser.get<float>("decay");
    bool do_test = parser.has("test");

    if (do_test)
        return dnn_test(json, saved);

    if (opt == "rms")
        return dnn_train<RMSprop>(json, saved, learn, decay, batch_size);
    if (opt == "adam")
        return dnn_train<adam>(json, saved, learn, decay, batch_size);
    if (opt == "adagrad")
        return dnn_train<adagrad>(json, saved, learn, decay, batch_size);
    if (opt == "momentum")
        return dnn_train<momentum>(json, saved, learn, decay, batch_size);
    return dnn_train<gradient_descent>(json, saved, learn, decay, batch_size);
}
