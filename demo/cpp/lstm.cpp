#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/dnn/layer.details.hpp>  // CV_DNN_REGISTER_LAYER_CLASS

using namespace cv;
using namespace cv::dnn;
using namespace std;

int batch = 5;
int features = 4;
int hidden = 3;
int seq_len = 2;

void runLayer(Ptr<Layer> layer, std::vector<Mat> &inpBlobs, std::vector<Mat> &outBlobs)
{
    size_t ninputs = inpBlobs.size();
    std::vector<Mat> inp(ninputs), outp, intp;
    std::vector<MatShape> inputs, outputs, internals;

    for (size_t i = 0; i < ninputs; i++)
    {
        inp[i] = inpBlobs[i].clone();
        inputs.push_back(shape(inp[i]));
    }

    layer->getMemoryShapes(inputs, 0, outputs, internals);
    for (size_t i = 0; i < outputs.size(); i++)
    {
        outp.push_back(Mat(outputs[i], CV_32F));
    }
    for (size_t i = 0; i < internals.size(); i++)
    {
        intp.push_back(Mat(internals[i], CV_32F));
    }

    layer->finalize(inp, outp);
    layer->forward(inp, outp, intp);

    size_t noutputs = outp.size();
    outBlobs.resize(noutputs);
    for (size_t i = 0; i < noutputs; i++)
        outBlobs[i] = outp[i];
}

int main(int argc, char** argv) {
	MatShape in_sz {1, seq_len, batch, features};
	MatShape out_sz {1, 1, batch, hidden};
	Mat out(4,out_sz.data(),CV_32F);
	Mat_<float> input(4,in_sz.data());
	input << 0.1467,  1.4530, -0.4364, -0.0212,
         -0.4735, -1.4737, -1.4300, -1.1509,
         -2.3083,  0.8632,  0.3320, -2.4869,
         -0.3625,  0.6295,  0.5539, -0.6521,
          2.1668,  0.6610, -1.2080, -0.4785,

          0.3470, -0.0960, -0.9315, -0.9260,
         -0.5117, -1.5838,  0.4338,  0.3337,
         -1.7227,  1.2616, -0.5277,  0.3209,
         -0.3351,  1.4093,  0.7629, -0.1848,
         -1.7194, -1.7030, -0.8111,  0.3194;


    Mat Wh, Wx, b;

    int numInp = total(in_sz);
    int numOut = total(out_sz);

	cout << "inp " << numInp << endl;
	cout << "out " << numOut << endl;

    Wh = Mat::zeros(4 * numOut, numOut, CV_32F);
    Wx = Mat::zeros(4 * numOut, numInp, CV_32F);
    b  = Mat::zeros(4 * numOut, 1, CV_32F);
	float k = sqrt(1.0/(4*numOut*numOut));
	cout << "k " << k << endl;
	randu(Wh,-k,k);
	randu(Wx,-k,k);
	randu(b, -k,k);

    LayerParams lp;
    lp.blobs.resize(3);
    lp.blobs[0] = Wh;
    lp.blobs[1] = Wx;
    lp.blobs[2] = b;
    lp.set<bool>("produce_cell_output", false);
    lp.set<bool>("use_timestamp_dim", false);
    Ptr<LSTMLayer> layer = LSTMLayer::create(lp);

	//MatShape outShape_ {out.size[0],out.size[1],out.size[2],out.size[3]};
    //layer->setOutShape(out_sz);
    std::vector<Mat> inputs, outputs;
    inputs.push_back(input);
    //outputs.push_back(out);
	runLayer(layer, inputs, outputs);
	cout << outputs.size() << " " << outputs[0].size << endl;
	cout << outputs[0] << endl;

    return 0;
}


/*
# https://github.com/opencv/opencv/pull/16817
# https://colah.github.io/posts/2015-08-Understanding-LSTMs/
# https://colab.research.google.com/drive/1Ad1R2wir63AXCX2-2YVEqXq9BSjxDr5i

import torch, torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, features, hidden, batch, num_layers=1, bi=False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(features, hidden, num_layers, bidirectional=bi)
        sz=1
        if bi: sz=2
        self.h0 = torch.zeros(sz * num_layers, batch, hidden)
        self.c0 = torch.zeros(sz * num_layers, batch, hidden)

    def forward(self, x):
        return self.lstm(x, (self.h0, self.c0))[0]

batch = 5
features = 4
hidden = 3
seq_len = 2
input = torch.randn(seq_len, batch, features)
print(input)

def run(bi=False):
  lstm = LSTM(features, hidden, batch, 1, bi)
  out = lstm(input)
  o = out[0]
  h = out[1][0]
  c = out[1][1]
  print(o.shape)
  print(h.shape)
  print(c.shape)
  print("o",o)
  print("h",h)
  print("c",c)

run(False)
run(True)

tensor([[[ 0.1467,  1.4530, -0.4364, -0.0212],
         [-0.4735, -1.4737, -1.4300, -1.1509],
         [-2.3083,  0.8632,  0.3320, -2.4869],
         [-0.3625,  0.6295,  0.5539, -0.6521],
         [ 2.1668,  0.6610, -1.2080, -0.4785]],

        [[ 0.3470, -0.0960, -0.9315, -0.9260],
         [-0.5117, -1.5838,  0.4338,  0.3337],
         [-1.7227,  1.2616, -0.5277,  0.3209],
         [-0.3351,  1.4093,  0.7629, -0.1848],
         [-1.7194, -1.7030, -0.8111,  0.3194]]])
torch.Size([5, 3])
torch.Size([3])
torch.Size([3])
o tensor([[-1.3933e-01,  1.6879e-01,  6.6749e-02],
        [-4.7055e-01, -4.5274e-02,  1.9420e-04],
        [-1.5185e-01, -1.0649e-01,  3.8526e-02],
        [-2.1706e-01, -1.4511e-02,  4.5358e-02],
        [ 8.0523e-02,  2.2574e-01,  8.3092e-03]], grad_fn=<SelectBackward>)
h tensor([-0.3903,  0.1071,  0.0715], grad_fn=<SelectBackward>)
c tensor([-0.4963, -0.1061, -0.1445], grad_fn=<SelectBackward>)
torch.Size([5, 6])
torch.Size([6])
torch.Size([6])
o tensor([[ 0.0197,  0.0863, -0.2584,  0.0552, -0.1514,  0.1319],
        [-0.0622, -0.0597, -0.2369, -0.2420, -0.1816,  0.1082],
        [ 0.1227,  0.0011, -0.1537,  0.0946, -0.1282, -0.0940],
        [-0.0262,  0.0459, -0.1174,  0.1549, -0.1320, -0.0704],
        [-0.2517,  0.1073, -0.4865, -0.0409, -0.3030,  0.3235]],
       grad_fn=<SelectBackward>)
h tensor([-0.1021,  0.1123, -0.5251, -0.0830, -0.2328,  0.1371],
       grad_fn=<SelectBackward>)
c tensor([-0.1761, -0.0957, -0.0314, -0.1594,  0.0296,  0.0709],
       grad_fn=<SelectBackward>)

*/