int loadClass(const String &dir, int label, Mat &trainData, Mat &trainLabels, const Size &siz) {
	vector<String> files;
	glob(dir, files);
	for (size_t i=0; i<files.size(); i++) {
		Mat img = imread(files[i], 0);
		if (img.empty()) continue;
		resize(img, img, siz);
		img.convertTo(img, CV_32F);

		trainData.push_back(img.reshape(1,1)); // add it as a single, flat row
		trainLabels.push_back(label);
	}
	return (int)trainLabels.size();
}

...

Mat trainData, trainLabels;
Size siz(60,60); // 3600 elements
int child  = loadClass("./images/child/*.png", 0, trainData, trainLabels, siz);
int adult  = loadClass("./images/adult/*.png", 1, trainData, trainLabels, siz);
int senior = loadClass("./images/senior/*.png", 2, trainData, trainLabels, siz);

model->train(ml::TrainData::create(trainData, ml::ROW_SAMPLES, trainLabels));

