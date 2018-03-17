#include <opencv2/xphoto/white_balance.hpp>
#include <opencv2/xphoto.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    int num_trees=-17, num_tree_nodes=-19;
    Mat feature_idx_Mat, thresh_vals_Mat, leaf_vals_Mat;
            FileStorage fs("color_balance_model.yml", 0);
cout << fs.isOpened() << endl;
            fs["feature_idx"] >> feature_idx_Mat;
            fs["thresh_vals"] >> thresh_vals_Mat;
            fs["leaf_vals"] >> leaf_vals_Mat;
            num_trees = fs["num_trees"];
            num_tree_nodes = fs["num_tree_nodes"];
            FileNode f1 = fs["num_trees"];
            cout << "isInt " << f1.isInt() << " " << f1.type() << endl;

cout << num_trees << " " << num_tree_nodes << endl;
cout << feature_idx_Mat.size() << endl;
cout << thresh_vals_Mat.size() << endl;
cout << leaf_vals_Mat.size() << endl;

    Mat img, img_out;
    img = imread("img/chili.jpg", CV_LOAD_IMAGE_COLOR);

    Ptr<xphoto::LearningBasedWB> wb;
    wb = xphoto::createLearningBasedWB();//"color_balance_model.yml");
    wb->balanceWhite(img, img_out);

    imshow( "Input", img );
    imshow( "Output", img_out );                   // Show our image inside it.

    waitKey(0);
    return 0;
}
