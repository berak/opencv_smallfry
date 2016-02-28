//
// https://iamtrask.github.io//2015/11/15/anyone-can-code-lstm/
// slightly changed from learning bits to more general feature-batch learning
//
// ph_16_02_28
//

#include "opencv2/opencv.hpp"
using namespace cv;

#include<iostream>
using namespace std;


namespace nn
{

    static Mat sigmoid(const Mat &x)
    {
        Mat m;
        cv::exp(Mat(-x), m);
        return 1.0/(1.0 + m);
    }

    static Mat sigmoid_deriv(const Mat &output)
    {
        Mat o;
        multiply(output,1-output,o);
        return o;
    }
    
    struct Rnn
    {
        Mat synapse_0, synapse_1, synapse_h;       
        int num_hidden() const { return synapse_0.cols; }

        void init(int i, int h, int o)
        {
            synapse_0 = Mat(i,h,CV_32F);
            synapse_1 = Mat(h,o,CV_32F);
            synapse_h = Mat(h,h,CV_32F);
            randu(synapse_0, -1,1);
            randu(synapse_1, -1,1);
            randu(synapse_h, -1,1);
        }

        double train_batch(const vector<Mat> &input, const vector<Mat> &output, int nhidden=16, float alpha = 0.1f)
        {
            if (synapse_0.empty())
                init(input[0].cols, nhidden, output[0].cols);

            vector<Mat> err_output;
            vector<Mat> prev_hidden(1, Mat::zeros(1, nhidden, CV_32F));
            for (size_t i=0; i<input.size(); i++)
            {
                const Mat &data   = input[i];
                const Mat &expect = output[i]; 
                const Mat &last_hidden = prev_hidden.back();

                Mat hidden = sigmoid((data * synapse_0).mul(last_hidden * synapse_h));
                prev_hidden.push_back(hidden);

                Mat result = sigmoid(hidden * synapse_1);
                err_output.push_back((expect - result).mul(sigmoid_deriv(result)));
            }

            int sH = prev_hidden.size();
            Mat delta(1, nhidden, CV_32F, 0.0f);
            Mat synapse_0_update(synapse_0.size(), CV_32F, 0.0f),
                synapse_1_update(synapse_1.size(), CV_32F, 0.0f),
                synapse_h_update(synapse_h.size(), CV_32F, 0.0f);

            Mat synapse_ht = synapse_h.t();
            Mat synapse_1t = synapse_1.t();

            double error = 0;
            for (size_t i=0; i<input.size(); i++)
            {
                int last = sH - i - 1;
                const Mat &data   = input[i];
                const Mat &hidden = prev_hidden[last];
                const Mat &last_hidden = prev_hidden[last - 1];
                const Mat &err = err_output[last-1];
                Mat hidden_delta = (delta * synapse_ht + err * synapse_1t).mul(sigmoid_deriv(hidden));
                delta = hidden;

                synapse_1_update += hidden.t() * err;
                synapse_h_update += last_hidden.t() * hidden_delta;
                synapse_0_update += data.t() * hidden_delta;
                error += sum(err)[0];
            }
            synapse_0 += synapse_0_update * alpha;
            synapse_1 += synapse_1_update * alpha;
            synapse_h += synapse_h_update * alpha;
            return error;
        }

        double train(const vector<Mat> &input, const vector<Mat> &output, int batchSize, int ngen, int nhidden=16, float alpha = 0.1f, float minerr=1.0e-6)
        {
            double err=0;
            int g=0;
            for (; g<ngen; g++)
            {
                vector<Mat> batch_in, batch_out;
                for (int b=0; b<batchSize; b++)
                {
                    int id = theRNG().uniform(0, int(input.size()));
                    batch_in.push_back(input[id]);
                    batch_out.push_back(output[id]);
                }
                err = train_batch(batch_in, batch_out, nhidden, alpha);
                if (err < minerr) break;
                if (g % 100 != 0) continue;
                cerr << format("gen   %5d err: %3.6f", g, err)<< endl;
            }
            cerr << format("final %5d err: %3.6f", g, err)<< endl;
            return err;
        }
    };
}

int main(int argc, char **argv)
{
    vector<Mat> input,output;
    input.push_back((Mat_<float>(1,4)<<1,2,3,4));    output.push_back(Mat_<float>(1,1)<<1);
    input.push_back((Mat_<float>(1,4)<<3,2,3,4));    output.push_back(Mat_<float>(1,1)<<1);
    input.push_back((Mat_<float>(1,4)<<1,2,3,4));    output.push_back(Mat_<float>(1,1)<<1);
    input.push_back((Mat_<float>(1,4)<<4,3,2,1));    output.push_back(Mat_<float>(1,1)<<2);
    input.push_back((Mat_<float>(1,4)<<4,3,2,2));    output.push_back(Mat_<float>(1,1)<<2);
    input.push_back((Mat_<float>(1,4)<<4,2,3,1));    output.push_back(Mat_<float>(1,1)<<2);

    nn::Rnn net;
    net.train(input, output, 4, 1000, 60, 0.2f);
    return 0;
}
