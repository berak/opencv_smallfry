#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using namespace std;

static Size siz(24,24); // @#Ö global..

// trainset:
// http://parnec.nuaa.edu.cn/xtan/data/datasets/dataset_B_Eye_Images.rar

void compute_hog_dir(const String &path, Mat &gradients, const Size & size)
{
    HOGDescriptor hog;
    hog.winSize = size;
    vector< Point > location;

    vector<String> fn;
    glob(path,fn);

    for(size_t i=0; i<fn.size(); i++)
    {
        Mat gray = imread(fn[i], 0);
        if (gray.empty())
            continue;
        vector< float > descriptors;
        hog.compute(gray, descriptors, Size( 8, 8 ), Size( 0, 0 ), location);
        gradients.push_back(Mat(descriptors).clone().reshape(1,1));
    }
}

void train_one_side(const String &pos, const String &neg, const String &xml)
{
    Mat grads;

    // compute pos/neg hogs:
    compute_hog_dir(pos, grads, siz);
    int npos = grads.rows;

    compute_hog_dir(neg, grads, siz);
    int nneg = grads.rows - npos;

    cerr << npos << "\t" << pos << endl;
    cerr << nneg << "\t" << neg << endl;

    // setup svm labels (1 for open, -1 for closed):
    Mat labels(grads.rows, 1, CV_32S, Scalar(-1));
    labels(Range(0,npos), Range::all()) = 1;

    // train svm and save:
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::NU_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setNu(0.5);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, 1e-6));
    svm->train(ml::TrainData::create(grads, ml::ROW_SAMPLE, labels));
    svm->save(xml);
}


//http://cmp.felk.cvut.cz/~uricamic/flandmark/
#include "flandmark/flandmark_detector.h"
enum // "mirrored" to http://cmp.felk.cvut.cz/~uricamic/flandmark/img/graph_constraints.png
{
    FACE_CENTER,
    LEFT_EYE_INNER,
    RIGHT_EYE_INNER,
    LEFT_MOUTH_CORNER,
    RIGHT_MOUTH_CORNER,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    NOSE
};


int main()
{
    // UNCOMMENT FOR ACTUAL TRAINING !:
    //train_one_side("E:/MEDIA/eyes/openLeftEyes/*.jpg", "E:/MEDIA/eyes/closedLeftEyes/*.jpg", "eyesLeft.xml");
    //train_one_side("E:/MEDIA/eyes/openRightEyes/*.jpg","E:/MEDIA/eyes/closedRightEyes/*.jpg","eyesRight.xml");

    namedWindow("flandmark eyes", 0);

    // load previously trained svm:
    Ptr<ml::SVM> svm_l = Algorithm::load<ml::SVM>("eyesLeft.xml");
    Ptr<ml::SVM> svm_r = Algorithm::load<ml::SVM>("eyesRight.xml");

    FLANDMARK_Model * model = flandmark_init("flandmark/flandmark_model.dat");

    // testset , cropped from tv images via face-cascade, resized to 90x90
    vector<String> fn;
    glob("E:/MEDIA/faces/tv10/*.png",fn);
    for (size_t i=0; i<fn.size(); i++)
    {
        Mat gray = imread(fn[i], 0);
        if (gray.empty())
            continue;
        
        // flandmark unfortunately expects an (extra) border around a given face roi:
        int width = gray.cols;
        int height = gray.rows;
        int B = 10;
        copyMakeBorder(gray,gray, B,B,B,B, BORDER_CONSTANT);

        // locate eye regions using flandmark:
        int npoints = model->data.options.M;
        int bbox[4] = { B, B, B + width, B + height };
        double *points = new double[2 * npoints];
        if(flandmark_detect(gray, bbox, model, points, 0) > 0)
        {
            cerr << "no landmarks found for " << fn[i] << endl;
	        continue;
        }
        vector<Point2d> landmarks;
        for (int j=0; j<npoints; j++)
        {
	        landmarks.push_back(Point2d(points[2 * j], points[2 * j + 1]));
        }
        delete[] points;

        // derive eye-rects from eye-landmarks:
        int off = 2; // the rect is a bit too small, make it slightly larger.
        int midy_l = (landmarks[LEFT_EYE_OUTER].y  + landmarks[LEFT_EYE_INNER].y)  / 2;
        int midy_r = (landmarks[RIGHT_EYE_OUTER].y + landmarks[RIGHT_EYE_INNER].y) / 2;
        int midx_l = (landmarks[LEFT_EYE_OUTER].x  + landmarks[LEFT_EYE_INNER].x)  / 2;
        int midx_r = (landmarks[RIGHT_EYE_OUTER].x + landmarks[RIGHT_EYE_INNER].x) / 2;
        int dist_l = landmarks[LEFT_EYE_INNER].x   - landmarks[LEFT_EYE_OUTER].x;
        int dist_r = landmarks[RIGHT_EYE_OUTER].x  - landmarks[RIGHT_EYE_INNER].x;
        Rect eye_l = Rect(landmarks[LEFT_EYE_OUTER].x-off,  midy_l - dist_l/2-off, dist_l+2*off, dist_l+2*off);
        Rect eye_r = Rect(landmarks[RIGHT_EYE_INNER].x-off, midy_r - dist_r/2-off, dist_r+2*off, dist_r+2*off);

        // make hog-features, and predict via SVM:
        vector< Point > location;
        vector< float > descriptors;
        HOGDescriptor hog;
        hog.winSize = siz;

        // left:
        Mat res_l;
        resize(gray(eye_l), res_l, siz);
        hog.compute(res_l, descriptors, Size( 8, 8 ), Size( 0, 0 ), location);
        Mat grad_l = Mat(descriptors,true).reshape(1,1);
        float pred_l = svm_l->predict(grad_l);

        // right:
        Mat res_r;
        resize(gray(eye_r), res_r, siz);
        hog.compute(res_r, descriptors, Size( 8, 8 ), Size( 0, 0 ), location);
        Mat grad_r = Mat(descriptors,true).reshape(1,1);
        float pred_r = svm_r->predict(grad_r);

        // draw stuff:
        cerr << pred_l << " " << pred_r << endl;
        if (pred_l > 0)
            circle(gray,Point(midx_l,midy_l),8,Scalar(0xff),1,LINE_AA);
        else
            rectangle(gray,eye_l,Scalar(0x00));

        if (pred_r > 0)
            circle(gray,Point(midx_r,midy_r),8,Scalar(0xff),1,LINE_AA);
        else
            rectangle(gray,eye_r,Scalar(0x00));

        imshow("flandmark eyes", gray);
        if (waitKey() == 27) break;
    }

    flandmark_free(model);
    return 0;
}
