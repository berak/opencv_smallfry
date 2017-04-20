#include "svm.h"
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class Model {
	static void silent(const char *) {}; // silence libsvm training debug output
	static svm_node *make_node(const Mat &A, int r=0) { // for a single row opencv feature
	    const float *dataPtr = A.ptr<float>(r); // Get data from OpenCV Mat
		svm_node *x_space = Malloc(svm_node, A.cols+1); // one more for the terminator
	    for (int c=0; c<A.cols; c++) {
	        x_space[c].index = c+1;  // Index starts from 1; Pre-computed kernel starts from 0
	        x_space[c].value = dataPtr[c];
	    }
	    x_space[A.cols].index = -1;  // End of sequence
	    return x_space;
	}
	svm_model *model;
	svm_problem prob;
public:

	Model(const String &filename) {
		model = svm_load_model(filename.c_str());
		CV_Assert(model != 0);
		prob.l=0;
	}

	Model(const Mat &A, const Mat &L) {
		svm_set_print_string_function(silent); // comment to see the debug output
		svm_parameter param = {0};
		param.svm_type = C_SVC;
		param.kernel_type = LINEAR;
		param.cache_size = 100;
		param.C = 1;
		param.eps = 1e-3;
		param.probability = 1;

	    prob.l = A.rows;
		prob.y = Malloc(double, prob.l);
		prob.x = Malloc(svm_node *, prob.l);
		for (int r=0; r<prob.l; r++) {
			prob.x[r] = make_node(A, r);
			prob.y[r] = L.at<int>(r, 0);
		}
		model = svm_train(&prob, &param);
	}

	~Model() {
		// fixme: if i ever *use* an svm_problem (it's ok with a model loaded from file),
		//   i can't release it after training, but have to keep it around
		//   for the lifetime of the model (s.a. svm.h:69), why ?
		if (prob.l) { // no need to do anything, if model was loaded from file
			for (int r=0; r<prob.l; r++) free(prob.x[r]);
			free(prob.x);
			free(prob.y);
		}
		svm_free_and_destroy_model(&model);
	}

	void probability(const Mat &query, Mat &result) {
		svm_node *x = make_node(query);
		double prob_est[2], prediction = svm_predict_probability(model, x, prob_est);
		result.push_back(prediction);
		result.push_back(prob_est[0]);
		result.push_back(prob_est[1]);
		free(x);
	}

	double predict(const Mat &query) { // a row sample
		svm_node *x = make_node(query);
		double prediction = svm_predict(model, x);
		free(x);
		return prediction;
	}

	bool save(const String &filename) {
		svm_save_model(filename.c_str(), model);
	}
};

int main(int argc, char **argv)
{	// demo, opencv like data
    Mat A = Mat::eye(5,5,CV_32F); // five very distinct vectors
	Mat L = (Mat_<int>(5,1) << 1,2,3,4,5); // each one is unique (multiclass)

	Model model(A,L); // from opencv data
	model.save("5x5.txt");
	//Model model("5x5.txt"); // or load from file (e.g. from svm-train)
	for (int r=A.rows-1; r>=0; r--) { // backwards, for a change ;)
		Mat result;
		model.probability(A.row(r), result); // prediction, probA, probB
		cerr << "prob " << r+1 << " " << result.t() << endl;
		double prediction = model.predict(A.row(r));
		cerr << "pred " << r+1 << " " << prediction << endl;
	}
	return 0;
}
