#ifndef __printnet_onboard__
#define __printnet_onboard__

void printnet(cv::dnn::Net &net, int batch_size, int channels, int H, int W) {
    cv::dnn::MatShape ms1 = { batch_size, channels, H, W };
    std::vector<cv::String> lnames = net.getLayerNames();
    for (size_t i=1; i<lnames.size()+1; i++) { // skip __NetInputLayer__
        cv::Ptr<cv::dnn::Layer> lyr = net.getLayer((unsigned)i);
        std::vector<cv::dnn::MatShape> in,out;
        net.getLayerShapes(ms1,i,in,out);
        std::cout << format("%-38s %-13s", lyr->name.c_str(), lyr->type.c_str());
        for (auto j:in) std::cout << "i" << cv::Mat(j).t() << "\t";
        for (auto j:out) std::cout << "o" << cv::Mat(j).t() << "\t";
        for (auto b:lyr->blobs) {              // what the net trains on, e.g. weights and bias
            std::cout << "b[" << b.size[0];
            for (size_t d=1; d<b.dims; d++) std::cout << ", " << b.size[d];
            std::cout << "]  ";
        }
    }
}

#endif // __printnet_onboard__
