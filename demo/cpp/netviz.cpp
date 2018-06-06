/*
File : main.cpp
Author : Rémi Ratajczak
E-mail : Remi.Ratjczak@gmail.com
License : MIT
This program demosntrates how to use the imagesFromBlob() method from the OpenCV dnn module.
This method returns a 2D array per image in the batch.
Each channel for each image equals the result of a filter activation.
*/

/* OpenCV things */
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>

/* Standard things */
#include <fstream>
#include <iostream>
#include <cstdlib>

/* Useful namespaces */
using namespace cv; //boo, remove it
using namespace cv::dnn; //boo, remove it
using namespace std; //boo, remove it


void visualizeInputsFromBlob(const cv::Mat& inputBlob, bool normalizeFlag, cv::Size size, double scaleFactor, cv::Scalar mean = cv::Scalar(-1,-1,-1))
{
	//A simple vector tha image (i.e. the result of each operation in the layer)
	std::vector<cv::Mat> vectorOfInputImagesFromBlob;

	std::cout << inputBlob.depth() << std::endl;
	//If the blob is not empty, extract images from it
	if (!inputBlob.empty())
		dnn::imagesFromBlob(inputBlob, vectorOfInputImagesFromBlob);

	//Try to visualize
	//The image should be fairly similar to the inputImg
	for (auto img : vectorOfInputImagesFromBlob)
	{
		//Inverse operations made by blobFromImage* metho
		if(scaleFactor != 0) img = img / scaleFactor;
		if(img.channels() == 3 && mean != cv::Scalar(-1, -1, -1)) img += mean;
		if(img.size() != size && size.height > 0 && size.width > 0) cv::resize(img, img, size);
		if(normalizeFlag) cv::normalize(img, img, 0, 1, cv::NORM_MINMAX);

		//Display
		cv::imshow("image_", img);
		cv::waitKey(0);

	}//blobs
}

void visualizAllChannelsFromAllBlobsInNet(Net& net, bool normalizeFlag, cv::Mat inputBlob, cv::Size size, double scaleFactor, cv::Scalar mean = cv::Scalar(-1))
{
	//For each layer in the network, we are going to perform a forward pass, retrieve the output blobs and extract the images from them
	for (string layer : net.getLayerNames())
	{
		//A container for our blobs
		std::vector<cv::Mat> vectorOfBlobs;

		//Set the network input - with GoogleNet, "data" is the name of the input layer
		net.setInput(inputBlob, "data");

		//Operate a forward pass, output the result of the selected layer (a blob)
		net.forward(vectorOfBlobs, layer);

		//For each blobs in our vectorOfBlobs
		for (cv::Mat blob : vectorOfBlobs)
		{
			//A vector to store the images (i.e. the result of each filter in the layer)
			std::vector<cv::Mat> vectorOfImages;

			//If the blob is not empty, extract images from it
			if (!blob.empty()) dnn::imagesFromBlob(blob, vectorOfImages);

			//Quality check
			for (auto image : vectorOfImages)
			{
				//Display useful data
				std::cout << "nbOfImages : " << vectorOfImages.size() << std::endl;
				std::cout << "channels : " << image.channels() << std::endl;
				std::cout << "size : " << image.size() << std::endl;

				//The channels are obtained using the split method
				//Display each 2D channel contained in the current image
				std::vector< cv::Mat > channels;
				cv::split(image, channels);
				for (auto channel : channels)
				{
					//Inverse operations made by blobFromImage* methods
					if (channel.size() != size && size.height > 0 && size.width > 0) cv::resize(channel, channel, size);
					if (scaleFactor != 0) channel = channel / scaleFactor;
					if (channel.channels() == 1 && mean != cv::Scalar(-1)) channel += mean;
					if (normalizeFlag) cv::normalize(channel, channel, 0, 1, cv::NORM_MINMAX);

					//Convert in CV_8U for Jet colormap
					channel = channel * 255;
					channel.convertTo(channel, CV_8U); //float to unsigned char for applyColorMap only
					cv::cvtColor(channel, channel, cv::COLOR_GRAY2BGR);
					cv::applyColorMap(channel, channel, cv::COLORMAP_JET);

					//Display
                    cv::imwrite("../"+layer + ".jpg", channel);
					cv::imshow(layer + " : output", channel);
					cv::waitKey(0);

				}//for loop on channels
			}//for loop on images

			 //Destroy bothering windows from Quality check
			cv::destroyAllWindows();

		} //for loop on blobs
	} //for loop on layers
}

int main(int argc, char **argv)
{
	//Load the model parameters paths in memory
	//You will find the caffemodel there: http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
	String modelTxt = "c:\\data\\mdl\\bvlc_googlenet.prototxt"; //definition of the model
	String modelBin = "c:\\data\\mdl\\bvlc_googlenet.caffemodel"; //weights of the model
	String imageFile = "img\\rubberwhale2.jpg"; //image to read - you can use your own
	String imageFile2 = "out.jpg"; //image to read - you can use your own
	String classNameFile = "c:\\data\\mdl\\synset_words.txt";//used for classification only - not presented here

	//Try to instantiate the network with its parameters
	Net net;
	try {
		net = dnn::readNetFromCaffe(modelTxt, modelBin);
	}
	catch (const cv::Exception& e) {
		std::cerr << "Exception: " << e.what() << std::endl;
		if (net.empty())
		{
			std::cerr << "Can't load network by using the following files: " << std::endl;
			std::cerr << "prototxt:   " << modelTxt << std::endl;
			std::cerr << "caffemodel: " << modelBin << std::endl;
			std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
			std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
			cv::waitKey(0);
			exit(-1);
		}
	}

	//Read an image
	Mat img = imread(imageFile);
	if (img.empty())
	{
		std::cerr << "Can't read image from the file: " << imageFile << std::endl;
		exit(-1);
	}

	// Put the image in a vector
	std::vector< cv::Mat > vectorOfInputImages;
	vectorOfInputImages.push_back(img);

	//Convert the image into a blob so that we could feed the network with it.
	//The blob will internally store the image in floating point precision (CV_32F).
	Mat inputBlob = blobFromImages(vectorOfInputImages, //the images to add to the blob
		                           1.0f, //a multiplicative factor
		                           Size(224, 224), //the size of the image in the blob / should correspond to the expected input size of the network / either crop/bilinear resizing can be used
		                           Scalar(104, 117, 123), //the mean of the model / in practice you should calculate it from your dataset / it is used to mean center the images
		                           false); //a boolean to convert a BGR image (default in OpenCV) to a RGB image / the format should correspond to the one used to train you network!

	//Visualize the inputs
	//gather the input Mat for the inputBlob, it should looks like (i.e. being equal to) the the original after normalization
	visualizeInputsFromBlob(inputBlob, true, img.size(), 1.0f, cv::Scalar(104, 117, 123));

	//Visualize the ouput of each layer
	visualizAllChannelsFromAllBlobsInNet(net, true, inputBlob, img.size(), -1, cv::Scalar(-1));

	return 0;
} //main
