#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include "bacf.hpp"

using namespace cv;
using namespace std;

struct  Params {

	//filter settings
	cv::Size model_sz = cv::Size(50, 50);
	float target_padding = 2.0;
	//learning parameters
	float update_rate = 0.003;
	float sigma_factor = 1.0 / 16.0;

	//scale settings
	float scale_step = 1.05;
	int num_scales = 1;

	float threshold = 0.03f;
}p;

struct TrackedRegion {
	TrackedRegion() { }
	TrackedRegion(const cv::Point2i init_center, const cv::Size init_size) : center(init_center), size(init_size) { }
	TrackedRegion(const cv::Rect box) : center(box.x + round((float)box.size().width / 2.0),
		box.y + round((float)box.size().height / 2.0)),
		size(box.size()) { }

	cv::Rect Rect() const {
		return cv::Rect(center.x - floor((float)size.width / 2.0),
			center.y - floor((float)size.height / 2.0),
			size.width, size.height);
	}

	TrackedRegion resize(const float factor) const {
		TrackedRegion newRegion;
		newRegion.center = center;
		newRegion.size = cv::Size(round(size.width *factor),
			round(size.height*factor));
		return newRegion;
	}

	cv::Point2i center;
	cv::Size size;
};


class BACF : public Tracker {
public:

	virtual bool initImpl(const Mat& image, const Rect2d& boundingBox);
	virtual bool updateImpl(const Mat& image, Rect2d& boundingBox);
 	virtual void write( FileStorage& fs ) const {}
 	virtual void read( const FileNode& fn ){}
private:
	//internal functions
	std::vector<cv::Mat> fft2(const cv::Mat featureData);

	int shift_index(const int index, const int length) const;
	cv::Mat make_labels(const cv::Size matrix_size, const cv::Size target_size, const float sigma_factor) const;

	void compute_ADMM();

	cv::Mat compute_response(const std::vector<cv::Mat>& filter, const std::vector<cv::Mat>& sample);

	std::vector<cv::Mat> compute_feature_vec(const cv::Mat& patch);
	void update_impl(const cv::Mat& image, const TrackedRegion& region, const float update_rate);
	cv::Mat channelMultiply(std::vector<cv::Mat> a, std::vector<cv::Mat> b, int flags, bool conjb);
	cv::Mat extractTrackedRegion(const cv::Mat image, const TrackedRegion region, const cv::Size output_sz);
	cv::Mat extractTrackedRegionSpec(cv::Mat model, const cv::Size output_sz);
	//parameters set on construction

	//internal state variables
	cv::Mat labelsf; //label function
	cv::Mat window; //cos (hann) window
	std::vector<cv::Mat> model_xf; //regularization marix
	std::vector<cv::Mat> filterf;

	float scale_factor;
	TrackedRegion target;

}; //end definition

struct DummyModel : TrackerModel
{
     virtual void modelUpdateImpl() CV_OVERRIDE {}
     virtual void modelEstimationImpl( const std::vector<Mat>& ) CV_OVERRIDE {}
};

bool BACF::initImpl(const Mat& image, const Rect2d& boundingBox) {
    model = makePtr<DummyModel>();
	//convert region into internal representation defined from center pixel and size
	//including the padding
	target = TrackedRegion(boundingBox).resize(p.target_padding);

	scale_factor = sqrt((float)target.size.area() / (float)p.model_sz.area());

	float resize_factor = (1.0 / scale_factor) * (1.0 / p.target_padding);
	TrackedRegion resized_target = target.resize(resize_factor);

	//create labels
	labelsf = make_labels(p.model_sz, resized_target.size, p.sigma_factor);

	//create window function
	cv::createHanningWindow(window, p.model_sz, CV_32FC1);

	//create the initial filter from the init patch
	update_impl(image, target, 1.0);
	return true;
}

bool BACF::updateImpl(const Mat& image, Rect2d& boundingBox) {
	cv::Mat pixels = extractTrackedRegion(image, target, p.model_sz);

	std::vector<cv::Mat> feature_vecf = compute_feature_vec(pixels);

	cv::Mat response = compute_response(filterf, feature_vecf);
	//	cout << response;

	cv::Point2i maxpos;
	double maxVal;
	//resp_newton
	cv::minMaxLoc(response, NULL, &maxVal, NULL, &maxpos);
	//cout << maxVal << endl;
	if (maxVal < p.threshold)
		return false;

	cv::Point2i translation(round(shift_index(maxpos.x, response.cols)*scale_factor),
		round(shift_index(maxpos.y, response.rows)*scale_factor));

	target.center = target.center + translation;

	TrackedRegion return_bb = target.resize(1.0 / p.target_padding);
	boundingBox = return_bb.Rect();

	update_impl(image, target, 2);
	return true;
}


//private functions
std::vector<cv::Mat> BACF::compute_feature_vec(const cv::Mat& patch) {
	//convert the data type to float
	cv::Mat feature_data;

	patch.convertTo(feature_data, CV_32FC1, 1.0 / 255.0, -0.5);

	std::vector<cv::Mat> feature_vec = fft2(feature_data);

	return feature_vec;
}


cv::Mat BACF::extractTrackedRegion(const cv::Mat image, const TrackedRegion region, const cv::Size output_sz) {
	int xMin = region.center.x - floor(((float)region.size.width) / 2.0);
	int yMin = region.center.y - floor(((float)region.size.height) / 2.0);

	int xMax = xMin + region.size.width;
	int yMax = yMin + region.size.height;

	int xMinPad, xMaxPad, yMinPad, yMaxPad;

	if (xMin < 0) {
		xMinPad = -xMin;
	}
	else {
		xMinPad = 0;
	}

	if (xMax > image.size().width) {
		xMaxPad = xMax - image.size().width;
	}
	else {
		xMaxPad = 0;
	}

	if (yMin < 0) {
		yMinPad = -yMin;
	}
	else {
		yMinPad = 0;
	}

	if (yMax > image.size().height) {
		yMaxPad = yMax - image.size().height;
	}
	else {
		yMaxPad = 0;
	}

	//compute the acual rectangle we will extract from the image
	cv::Rect extractionRegion = cv::Rect(xMin + xMinPad,
		yMin + yMinPad,
		(xMax - xMin) - xMaxPad - xMinPad,
		(yMax - yMin) - yMaxPad - yMinPad);

	//make sure the patch is not completely outside the image
	if (extractionRegion.x + extractionRegion.width > 0 &&
		extractionRegion.y + extractionRegion.height > 0 &&
		extractionRegion.x < image.cols &&
		extractionRegion.y < image.rows) {

		cv::Mat real_patch(region.size, image.type());


		//replicate along borders if needed
		if (xMinPad > 0 || xMaxPad > 0 || yMinPad > 0 || yMaxPad > 0) {
			cv::copyMakeBorder(image(extractionRegion), real_patch, yMinPad,
				yMaxPad, xMinPad, xMaxPad, cv::BORDER_REPLICATE);

		}
		else {
			real_patch = image(extractionRegion);
		}

		if (!(real_patch.size().width == region.size.width && real_patch.size().height == region.size.height)) {
			//cout << "kasst" << endl;
		}

		cv::Mat ds_patch;
		cv::resize(real_patch, ds_patch, output_sz);

		return ds_patch;

	}
	else {
		cv::Mat dummyRegion = cv::Mat::zeros(region.size, image.type());
		cv::Mat ds_patch;
		cv::resize(dummyRegion, ds_patch, output_sz);

		return ds_patch;
	}
}
void BACF::update_impl(const cv::Mat& image, const TrackedRegion& region, const float frame) {
	//extract pixels to use for update
	cv::Mat pixels = extractTrackedRegion(image, region, p.model_sz);
	std::vector<cv::Mat> feature_vecf = compute_feature_vec(pixels);

	if (frame == 1.0) {
		model_xf = feature_vecf;
	}
	else {
		for (int i = 0; i<model_xf.size(); i++)
			model_xf[i] = ((1 - p.update_rate)*model_xf[i]) + (p.update_rate* feature_vecf[i]);
	}

	compute_ADMM();
}

cv::Mat BACF::channelMultiply(std::vector<cv::Mat> a, std::vector<cv::Mat> b, int flags, bool conjb) {
	cv::Mat prod;
	cv::Mat sum = cv::Mat::zeros(a[0].size(), a[0].type());
	for (unsigned int i = 0; i < a.size(); ++i) {
		cv::mulSpectrums(a[i], b[i], prod, flags, conjb);
		sum += prod;
	}
	return sum;
}


std::vector<cv::Mat> BACF::fft2(const cv::Mat featureData) {
	std::vector<cv::Mat> channels(featureData.channels());
	std::vector<cv::Mat> channelsf(featureData.channels());
	cv::split(featureData, channels);

	for (size_t i = 0; i < channels.size(); ++i) {
		cv::Mat windowed;
		cv::multiply(channels[i], window, windowed);
		cv::dft(windowed, channelsf[i], 0);
	}

	return channelsf;
}

int BACF::shift_index(const int index, const int length) const {
	int shifted_index;

	if (index > length / 2) {
		shifted_index = -length + index;
	}
	else {
		shifted_index = index;
	}

	return shifted_index;
}

cv::Mat BACF::make_labels(const cv::Size matrix_size, const cv::Size target_size, const float sigma_factor) const {
	cv::Mat new_labels(matrix_size.height, matrix_size.width, CV_32F);

	const float sigma = std::sqrt((float)target_size.area()) * sigma_factor;
	const float constant = -0.5 / pow(sigma, 2);

	for (int x = 0; x < matrix_size.width; x++) {
		for (int y = 0; y < matrix_size.height; y++) {
			int shift_x = shift_index(x, matrix_size.width);
			int shift_y = shift_index(y, matrix_size.height);
			float value = std::exp(constant*(std::pow(shift_x, 2) + std::pow(shift_y, 2)));
			new_labels.at<float>(y, x) = value;
		}
	}

	cv::Mat labels_dft;
	cv::dft(new_labels, labels_dft);
	return labels_dft;
}

void BACF::compute_ADMM() {
	std::vector<cv::Mat>l_f;
	std::vector<cv::Mat>h_f;
	int mu = 1;
	float T = (float)p.model_sz.area();
	cv::Mat S_xx = channelMultiply(model_xf, model_xf, 0, true);
	filterf.clear();
	for (int i = 0; i < 3; i++)
	{
		cv::Mat	lp = cv::Mat::zeros(model_xf[0].size(), model_xf[0].type());
		l_f.push_back(lp);
		cv::Mat	hp = cv::Mat::zeros(model_xf[0].size(), model_xf[0].type());
		h_f.push_back(hp);
		cv::Mat	fp = cv::Mat::zeros(model_xf[0].size(), model_xf[0].type());
		filterf.push_back(fp);
	}
	for (int i = 0; i < 2; i++)
	{
		cv::Mat B = S_xx + (T*mu);
		cv::Mat S_lx = channelMultiply(l_f, model_xf,0,true);
		cv::Mat S_hx = channelMultiply(h_f, model_xf,0,true);
		for (int j = 0; j<model_xf.size(); j++)
		{
			cv::Mat mlabelf, S_xxyf, mS_lx, mS_hx;
			cv::multiply(labelsf, model_xf[j], mlabelf);
			cv::multiply(S_xx, mlabelf, S_xxyf);
			cv::multiply(S_lx, model_xf[j], mS_lx);
			cv::multiply(S_hx, model_xf[j], mS_hx);
			cv::Mat h;
			cv::Mat ghj;
			cv::divide(S_xxyf.mul(1 / (T*mu)) - mS_lx.mul(1 / mu) + mS_hx,B,ghj);
			filterf[j] = ( mlabelf.mul(1/(T*mu)) -l_f[j].mul(1/mu)+ h_f[j]) - ghj;
			cv::dft((filterf[j].mul(mu) + l_f[j]), h, cv::DFT_INVERSE | cv::DFT_SCALE);
			cv::Mat t = extractTrackedRegionSpec(h.mul(1/mu), p.model_sz);
			cv::dft(t, h_f[j]);
			l_f[j] = l_f[j] + ((filterf[j].mul(mu) - h_f[j]));
		}
		mu = 10;
	}
}

cv::Mat BACF::extractTrackedRegionSpec(cv::Mat model, const cv::Size output_sz)
{
	cv::Rect r(output_sz.width / 4, output_sz.height / 4, output_sz.width / 2, output_sz.height / 2);
	TrackedRegion tr = r;
	cv::Mat	lp1 = cv::Mat::zeros(model.size(), model.type());
	cv::Mat	lp = extractTrackedRegion(model, tr, output_sz / 2);
	lp.copyTo(lp1(r));
	//std::cout << lp1;
	return lp1;
}

cv::Mat BACF::compute_response(const std::vector<cv::Mat>& filter, const std::vector<cv::Mat>& sample) {
	cv::Mat response;

	cv::Mat resp_dft = channelMultiply(filter, sample, 0, false);
	cv::dft(resp_dft, response, cv::DFT_INVERSE|cv::DFT_SCALE);
	//cout << response;
	return response;
}

Ptr<Tracker> createTrackerBACF() { return makePtr<BACF>(); }

#if 0 // standalone
int main(int argc, char** argv) {
	VideoCapture cap(0);
	Mat frame;
	cap >> frame;

	// get bounding box
	Rect2d roi = selectROI("tracker",frame,true,false);

	//quit if ROI was not selected
	if (roi.width == 0 || roi.height == 0)
		return 0;

	Ptr<Tracker> tracker = createTrackerBACF();
	tracker->init(frame, roi);

	// do the tracking
	printf("Start the tracking process, press ESC to quit.\n");
	for (;;) {
		// get frame from the video
		cap >> frame;

		// stop the program if no more images
		if (frame.rows == 0 || frame.cols == 0)
			break;

		// update the tracking result
		bool ok = tracker->update(frame, roi);
		if (ok) {
			// draw the tracked object
			rectangle(frame, roi, Scalar(255, 0, 0), 2, 1);
		} else {
			cout << "lost !" <<endl;
		}
		// show image with the tracked object
		imshow("tracker", frame);

		//quit on ESC button
		if (waitKey(1) == 27)break;
	}
}
#endif
