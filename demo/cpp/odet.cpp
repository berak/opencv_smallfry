
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
using namespace cv;
using namespace cv::ximgproc;
using std::vector;
using std::ifstream;
using std::ios;
using std::cerr;
using std::endl;

#include "profile.h"

//
// b g r y cr cb h s v like pearls ona string.
//
Mat feature_color(const Vec3b &pixel) {
    // normalized bgr;
    float b = float(pixel[0]) / 255;
    float g = float(pixel[1]) / 255;
    float r = float(pixel[2]) / 255;

    // ycrcb
    float y  = 0.299f * r + 0.587f * g + 0.114f * b;
    float cr = 0.713f * (r - y) + 0.5f;
    float cb = 0.564f * (b - y) + 0.5f;

    // hsv
    float M(max(max(b,g),r));
    float m(min(min(b,g),r));
    float md = M - m;
    float A = 1.0f / 6;
    float v =  M;
    float s = (M>0)   ? (md/M) : 0;
    float h = (md==0) ? 0 :
              (M==r)  ? (    A*(g-b)/md) :
              (M==g)  ? (2*A+A*(b-r)/md) :
                        (3*A+A*(r-g)/md);

    return Mat_<float>(1,9) << b,g,r, y,cr,cb, h,s,v; // 9x1
}


// CIFAR-10 binary version
struct Cifar10
{
    bool gray;

    Cifar10(bool gray=false)
        : gray(gray)
    {}

    //
    //In other words, the first byte is the label of the first image, which is a number in the range 0-9. The next 3072 bytes are the values of the pixels of the image. The first 1024 bytes are the red channel values, the next 1024 the green, and the final 1024 the blue. The values are stored in row-major order, so the first 32 bytes are the red channel values of the first row of the image.
    //Each file contains 10000 such 3073-byte "rows" of images, although there is nothing delimiting the rows. Therefore each file should be exactly 30730000 bytes long.
    //
    void read_batch(String filename, vector<Mat>& vec, vector<char> & lab)
    {
    	PROFILE;
        ifstream file (filename.c_str(), ios::binary);

        if (! file.is_open())
            CV_Error(0, String("could not open: ") + filename);

        for (int i=0; i<10000; i++)
        {
            char num;
            file.read((char*) &num, sizeof(char));
            lab.push_back(num);

            char planes[3][1024];
            file.read((char*) &planes[0], 1024*sizeof(char));
            file.read((char*) &planes[1], 1024*sizeof(char));
            file.read((char*) &planes[2], 1024*sizeof(char));

            Mat m;
            if (gray) // green only
            {
                m = Mat(32,32,CV_8U,planes[1]);
            }
            else
            {
                Mat p[3] = {
                    Mat(32,32,CV_8U,planes[0]),
                    Mat(32,32,CV_8U,planes[1]),
                    Mat(32,32,CV_8U,planes[2])
                };
                merge(p,3,m);
            }
            vec.push_back(m.clone());
        }
    }
};

#if 0

// returns column vec
Mat compute_hog(const Mat &gray, const Size & size)
{
	PROFILE;
    HOGDescriptor hog;
    hog.winSize = size;
    vector< Point > location;
    vector< float > descriptors;
    {
	    PROFILEX("hog.compute");
	    hog.compute(gray, descriptors, Size( 8, 8 ), Size( 0, 0 ), location);
    }
    return Mat(descriptors).clone();
}
Mat feature_hog (const Mat &img)
{
	PROFILE;
	//Mat fimg;
	//img.convertTo(fimg, CV_32F, 1.0f/255);
	Mat gray;
	if (img.channels() == 1)
		gray = img;
	else
		cvtColor(img, gray, COLOR_BGR2GRAY);

	Mat f = compute_hog(gray, Size(32,32));
	return f.reshape(1,1);
}


Mat feature_super (const Mat &img)
{
	PROFILE;
    int num_iterations = 2;
    int prior = 2;
    bool double_step = false;
    int num_superpixels = 16;
    int num_levels = 3;
    int num_histogram_bins = 5;

    Ptr<SuperpixelSEEDS> seeds;
    {
    	PROFILEX("seeds");
	    seeds = createSuperpixelSEEDS(
	    	img.rows, img.cols, img.channels(), num_superpixels,
	        num_levels, prior, num_histogram_bins, double_step);
    }

    Mat converted;
    cvtColor(img, converted, COLOR_BGR2HSV);
    {
    	PROFILEX("seeds:iterate");
    	seeds->iterate(converted, num_iterations);
	}

    Mat features;
    {
    	PROFILEX("seeds:features");
	    Mat labels;
	    seeds->getLabels(labels);
	    for (int i=0; i<seeds->getNumberOfSuperpixels(); i++)
	    {
	    	Scalar m = mean(img, (labels == i));
	    	for (int j=0; j<img.channels(); j++)
	    	{
	   			features.push_back(float(m[j]));
	    	}
	    }
	}
    return features.reshape(1,1);
}

#endif

Mat feature_grid(const Mat &img)
{
	PROFILE;
    int step = 4;
    Mat features;
    for (int r=0; r<img.rows; r+=step)
    {
	    for (int c=0; c<img.cols; c+=step)
	    {
            // sample mean color centers (9):
            Rect roi(c, r, step, step);
	    	{
	    		PROFILEX("grid:color");
		    	Scalar m = mean(img(roi));
		    	Vec3b col(m[0], m[1], m[2]);
		    	Mat fea = feature_color(col);
		    	features.push_back(fea.reshape(1, fea.total()));
		    }
            // sample neighbourhood gradients (channels*4)
		    {
		    	PROFILEX("grid:grads");
                int ds = step / 2;
		    	int cr = r + ds;
		    	int cc = c + ds;
                Vec3b g1 = img.at<Vec3b>(cr+ds,cc   ) - img.at<Vec3b>(cr-ds,cc   );
                Vec3b g2 = img.at<Vec3b>(cr+ds,cc+ds) - img.at<Vec3b>(cr-ds,cc-ds);
                Vec3b g3 = img.at<Vec3b>(cr,   cc+ds) - img.at<Vec3b>(cr,   cc-ds);
                Vec3b g4 = img.at<Vec3b>(cr+ds,cc+ds) - img.at<Vec3b>(cr-ds,cc-ds);

		 		//for (int i=0; i<img.channels(); i++)
		 		{
                    int i=1;
		 			/*float g1 = img.at<Vec3b>(cr+ds,cc   )[i] - img.at<Vec3b>(cr-ds,cc   )[i];
			 		float g2 = img.at<Vec3b>(cr+ds,cc+ds)[i] - img.at<Vec3b>(cr-ds,cc-ds)[i];
			 		float g3 = img.at<Vec3b>(cr,   cc+ds)[i] - img.at<Vec3b>(cr,   cc-ds)[i];
			 		float g4 = img.at<Vec3b>(cr+ds,cc+ds)[i] - img.at<Vec3b>(cr-ds,cc-ds)[i];
			    	*/
                    features.push_back(float(g1[i])/255.0f);
			    	features.push_back(float(g2[i])/255.0f);
			    	features.push_back(float(g3[i])/255.0f);
			    	features.push_back(float(g4[i])/255.0f);
			    }
            }
            /*
            // sample x-y projections (16+16):
		    {
		    	PROFILEX("grid:project");
                int N = 4;

                Mat imr;
		        resize(img(roi), imr, Size(N,N));

            	Mat px; reduce(imr, px, 1, REDUCE_SUM, CV_32F);
		    	Mat py; reduce(imr, py, 0, REDUCE_SUM, CV_32F);
		    	for (int i=0; i<N; i++)
		    	{
					features.push_back(px.at<float>(i));
					features.push_back(py.at<float>(i));
				}
		    }
            */
	    }
	}
    return features.reshape(1,1);
}

// preprocess cifar for svm:
void process(const vector<Mat> &dat_in, const vector<char> &lab_in,
	         Mat &dat_out, Mat &lab_out)
{
	PROFILE;
	for (size_t i=0; i<dat_in.size(); i++)
	{
		lab_out.push_back(int(lab_in[i]));
		dat_out.push_back(feature_grid(dat_in[i]));
	}
	cerr << "in  " << dat_in.size() << " items " << dat_in[0].size() << endl;
	cerr << "out " << dat_out.rows << " items " << dat_out.cols << " " << ((dat_out.total() * dat_out.elemSize()) / (1024*1024)) << " mb." << endl;
}



String path_cifar  = "c:/data/cifar10";          //!< http://www.cs.toronto.edu/~kriz/cifar.html

int train()
{
    PROFILE;

    Mat train_dat, train_lab;

    Cifar10 cifar(false);
    for (int i=1; i<=2; i++)
    {
        vector<Mat> cif_dat;
        vector<char> cif_lab;
        cifar.read_batch(path_cifar + format("/data_batch_%d.bin",i), cif_dat, cif_lab);
        process(cif_dat, cif_lab, train_dat, train_lab);
    }
    // train svm and save:
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::NU_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setNu(0.5);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, 1e-6));
    svm->train(ml::TrainData::create(train_dat, ml::ROW_SAMPLE, train_lab));
    svm->save("cifar.xml");

    return 0;
}

int test()
{
    PROFILE;

    Mat test_dat, test_lab;

    Cifar10 cifar(false);
    vector<Mat> cif_dat;
    vector<char> cif_lab;
    cifar.read_batch(path_cifar + "/test_batch.bin", cif_dat, cif_lab);
    process(cif_dat, cif_lab, test_dat, test_lab);

    Ptr<ml::SVM> svm = ml::SVM::load("cifar.xml");
    if (! svm->isTrained())
        return -1;

    Mat res;
    svm->predict(test_dat, res);
    res.convertTo(res, CV_32S);
    Mat ok = (res == test_lab);
    int good = countNonZero(ok);
    cerr << good << "/" << res.total() << " " << (float(good)/res.total()) << endl;
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc>1)
        return train();
    return test();
}
