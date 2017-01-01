
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;


typedef uchar BYTE;
int readFlippedInteger(FILE *fp)
{
    int ret = 0;

    BYTE *temp;

    temp = (BYTE *)(&ret);
    fread(&temp[3], sizeof(BYTE), 1, fp);
    fread(&temp[2], sizeof(BYTE), 1, fp);
    fread(&temp[1], sizeof(BYTE), 1, fp);

    fread(&temp[0], sizeof(BYTE), 1, fp);

    return ret;
}
Ptr<ml::KNearest> getKnn()
{
	Ptr<ml::KNearest> knn(ml::KNearest::create());

	FILE *fp = fopen("c:/data/mnist/train-images.idx3-ubyte", "rb");
	FILE *fp2 = fopen("c:/data/mnist/train-labels.idx1-ubyte", "rb");

	if (!fp || !fp2)
	{
	    cout << "can't open file" << endl;
	}

	int magicNumber = readFlippedInteger(fp);
	int numImages = readFlippedInteger(fp);
	int numRows = readFlippedInteger(fp);
	int numCols = readFlippedInteger(fp);
	fseek(fp2, 0x08, SEEK_SET);

	int size = numRows * numCols;

	cout << "size: " << size << endl;
	cout << "rows: " << numRows << endl;
	cout << "cols: " << numCols << endl;

	Mat_<float> trainFeatures(numImages, size);
	Mat_<int> trainLabels(1, numImages);

	BYTE *temp = new BYTE[size];
	BYTE tempClass = 0;
	for (int i = 0; i < numImages; i++)
	{
	    fread((void *)temp, size, 1, fp);
	    fread((void *)(&tempClass), sizeof(BYTE), 1, fp2);

	    trainLabels[0][i] = (int)tempClass;

	    for (int k = 0; k < size; k++)
	    {
	        trainFeatures[i][k] = (float)temp[k];
	    }
	}
printf("start training.\n");
	bool ok = knn->train(trainFeatures, ml::ROW_SAMPLE, trainLabels);
printf("trained %d.\n", ok);
	return knn;

}

//When I test the algorithm with the 10k images file MNIST provide I have: Accuracy: 96.910000 which is a good news :) The code to test the knn trained is here:

void testKnn(Ptr<ml::KNearest> knn, bool debug)
{
	int totalCorrect = 0;

	FILE *fp = fopen("c:/data/mnist/t10k-images.idx3-ubyte", "rb");
	FILE *fp2 = fopen("c:/data/mnist/t10k-labels.idx1-ubyte", "rb");

	int magicNumber = readFlippedInteger(fp);
	int numImages = readFlippedInteger(fp);
	int numRows = readFlippedInteger(fp);
	int numCols = readFlippedInteger(fp);
	fseek(fp2, 0x08, SEEK_SET);

	int size = numRows * numCols;

	Mat_<float> testFeatures(numImages, size);
	Mat_<int> expectedLabels(1, numImages);

	BYTE *temp = new BYTE[size];
	BYTE tempClass = 0;

	int K = 5;
	Mat response, dist, m;

	for (int i = 0; i < numImages; i++)
	{

	    if (i % 1000 == 0 && i != 0)
	    {
	        cout << i << endl;
	    }

	    fread((void *)temp, size, 1, fp);
	    fread((void *)(&tempClass), sizeof(BYTE), 1, fp2);

	    expectedLabels[0][i] = (int)tempClass;

	    for (int k = 0; k < size; k++)
	    {
	        testFeatures[i][k] = (float)temp[k];
	    }

	    // test to verify if createMatFromMNIST and createMatToMNIST are well.
	    m = testFeatures.row(i);

	    float p = knn->findNearest(m, K, noArray(), response, dist);

	    /*if (debug)
	    {
	        cout << "response: " << response << endl;
	        cout << "dist: " << dist << endl;
	        Mat m2 = createMatFromMNIST(m);
	        showImage(m2);
	        // Mat m3 = createMatToMNIST(m2);
	        // showImage(m3);
	    }*/

	    if (expectedLabels[0][i] == p)//response.at<float>(0))
	    {
	        totalCorrect++;
	    }
	}
	printf("Accuracy: %f ", (double)totalCorrect * 100 / (double)numImages);
}


int main(int argc, char** argv) {
	Ptr<ml::KNearest> knn = getKnn();
	testKnn(knn,false);
    return 0;
}
