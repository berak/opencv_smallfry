#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/layer.details.hpp>


class SwishLayer : public cv::dnn::Layer
{
public:
    SwishLayer(const cv::dnn::LayerParams &params) : Layer(params)
    {
    }

    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
    {
        return cv::Ptr<cv::dnn::Layer>(new SwishLayer(params));
    }

    virtual bool getMemoryShapes(const std::vector<std::vector<int> > &inputs,
                                 const int requiredOutputs,
                                 std::vector<std::vector<int> > &outputs,
                                 std::vector<std::vector<int> > &internals) const CV_OVERRIDE
    {
        CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
        std::vector<int> outShape(4);
        outShape[0] = inputs[0][0];  // batch size
        outShape[1] = inputs[0][1];  // number of channels
        outShape[2] = inputs[0][2];
        outShape[3] = inputs[0][3];
        outputs.assign(1, outShape);
        return false;
    }

    virtual void forward(cv::InputArrayOfArrays inputs_arr,
                         cv::OutputArrayOfArrays outputs_arr,
                         cv::OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {

        std::vector<cv::Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        cv::Mat& inp = inputs[0];
        cv::Mat& out = outputs[0];

        const int batchSize = inp.size[0];
        const int numChannels = inp.size[1];
        const int height = out.size[2];
        const int width = out.size[3];


        int sz[] = { (int)batchSize, numChannels, height, width };
        out.create(4, sz, CV_32F);
        /*for(int i=0; i<batchSize; i++)
        {
            for(int j=0; j<numChannels; j++)
            {
                cv::Mat plane(inp.size[2], inp.size[3], CV_32F, inp.ptr<float>(i,j));
                cv::Mat crop = plane(cv::Range(ystart,yend), cv::Range(xstart,xend));
                cv::Mat targ(height, width, CV_32F, out.ptr<float>(i,j));
                crop.copyTo(targ);
            }
        }*/
        inp.copyTo(out);
    }
};

// https://github.com/lukemelas/EfficientNet-PyTorch/issues/91#issuecomment-542994572
using namespace cv;
using namespace std;
int main(int argc, char** argv) {
    CV_DNN_REGISTER_LAYER_CLASS(Swish, SwishLayer);
    string folder = "c:/data/dnn/";
    dnn::Net net;
    //net = dnn::readNet(folder+"enetb0-coco_final.weights", folder+"efficientnet_b0.cfg");
    //net = dnn::readNet(folder+"efficientnet_b0.onnx");
    net = dnn::readNet(folder+"test-b0.onnx");
    Mat m(240,240,CV_8UC3);
    Mat b; dnn::blobFromImage(m,b);
    cout << b.size << endl;
    net.setInput(b);
    vector<string> names = net.getLayerNames();
    for (auto n:names)
    	cout << n << endl;
    net.dump();
    //net.dumpToFile("eff.txt");
    return 0;
}