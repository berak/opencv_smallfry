/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe © Wiley & Sons Inc. 2014.  All rights reserved.
 */
//#include "Utilities.h"
#include "opencv2/opencv.hpp"
#include "profile.h"
#include <fstream>

using namespace std;
using namespace cv;

// original idea
void ChamferMatching(const Mat& input, const Mat& model, Mat& matching)
{
	PROFILE;
	// Extract the model points (as they are sparse).
	std::vector<Point> model_points;
	int image_channels = model.channels();
	for (int model_row=0; (model_row < model.rows); model_row++)
	{
		for (int model_column=0; (model_column < model.cols); model_column++)
		{
			if (model.at<float>(model_row,model_column) > 0)
			{
				model_points.push_back(Point(model_column,model_row));
			}
		}
	}
	int num_model_points = model_points.size();
	int rows = input.rows - model.rows + 1;
	int cols = input.cols - model.cols + 1;
	cout << num_model_points << " model_points " << rows << " rows " << cols << " cols." << endl;
	// Try the model in every possible position
	matching = Mat(rows, cols, CV_32FC1);
	for (int search_row=0; search_row < rows; search_row++)
	{
		float *output_point = matching.ptr<float>(search_row);
		for (int search_column=0; search_column < cols; search_column++)
		{
			float matching_score = 0; //99999999999.9;
			for (int p=0; (p < num_model_points); p++) {
				int x = model_points[p].x + search_column;
				int y = model_points[p].y + search_row;
				float v = input.at<float>(y, x);
				matching_score += v;
				//matching_score = std::min(v, matching_score);
			}
			*output_point = matching_score;
			output_point++;
		}
	}
}

// the shortest, yet least effective one
void Chamfer2(const Mat& input, const Mat& model, Mat& matching)
{
	PROFILE;
	int rows = input.rows - model.rows + 1;
	int cols = input.cols - model.cols + 1;
	matching = Mat(rows, cols, CV_32FC1, 0.0f);
	for (int search_row=0; search_row < rows; search_row++)
	{
		for (int search_column=0; search_column < cols; search_column++)
		{
			Mat diff, roi(input, Rect(search_column, search_row, model.cols, model.rows));
			min(roi, model, diff);
			matching.at<float>(search_row,search_column) = sum(diff)[0] / diff.total();
		}
	}
}

// shortening #1 to 20 lines
void Chamfer3(const Mat& input, const Mat& model, Mat& matching)
{
	PROFILE;
	std::vector<Point> pts;
	findNonZero(model > 0, pts);

	int rows = input.rows - model.rows + 1;
	int cols = input.cols - model.cols + 1;
	Mat_<float> match(rows, cols, 0.0f), inp(input); // templated for convenience of access
	for (int r=0; r < rows; r++)
	{
		for (int c=0; c < cols; c++)
		{
			float score = 0;
			for (auto p:pts) {
				score += inp(p + Point(c,r));
			}
			match(r,c) = score;
		}
	}
	matching = match;
}


Mat proc(Mat M) {
	Mat edge_image;
	if (M.channels()>1) cvtColor(M,edge_image,COLOR_BGR2GRAY);
	else edge_image=M;
	Canny(edge_image, edge_image, 100, 200, 3);
	threshold( edge_image, edge_image, 127, 255, THRESH_BINARY );
	//distanceTransform( edge_image, edge_image, CV_DIST_L2, 3);

	edge_image.convertTo(edge_image, CV_32F);
	return edge_image;
}

float iou(const Rect &a, const Rect &b) {
    return float((a & b).area()) / (a | b).area();
}

int main(int argc, char **argv) {
	bool dodist=false;
	String match="ChamferMatching";
	for (int i=1; i<argc; i++) {
		if (argv[i][0] =='d') {
			dodist=true;
			continue;
		}
		if (argv[i][0] =='2') {
			match="Chamfer2";
			continue;
		}
		if (argv[i][0] =='3') {
			match="Chamfer3";
			continue;
		}
	}
	Rect ROIS[]{{283,106,34,31},{243,112,29,30},{269,162,34,19},{267,137,38,21},{21,52,47,51},{103,24,36,46}};
	Mat img = imread("img/oude-man.jpg");
	Mat edge = proc(img);
	Mat dx;
	if (dodist) {
		Mat _x;
		edge.convertTo(_x, CV_8U, 255);
		distanceTransform(_x, dx, CV_DIST_L2, 3);
		cout << "dx " << dx.type() << " " << dx.size() << endl;
	} else {
		dx = edge;
	}
	for (auto roi:ROIS) {
		Mat temp = edge(roi); // r eye
		Mat res;
		int64 t0 = getTickCount();
		if (match == "ChamferMatching")
			ChamferMatching(dx,temp,res);
		else if (match=="Chamfer2")
			Chamfer2(edge,temp,res);
		else
			Chamfer3(edge,temp,res);
		int64 t1 = getTickCount();
		cerr << match << " " << ((t1-t0)/getTickFrequency()) << " seconds." << endl;
		//cout << edge.type() << " " << temp.type() << " " << res.type() << endl;
		Point p;
		double m=0;
		minMaxLoc(res, 0,&m, 0,&p); // maximum
		float io = iou(roi, Rect(p,roi.size()));
		cout << p << " " << roi.tl() << " " << m << " " << io << endl;
		Rect r(p.x,p.y,temp.cols,temp.rows);
		rectangle(img,roi,Scalar(0,200,0),3);
		rectangle(img,r,Scalar(0,0,200),1);
		imshow("M",img);
		imshow("E",edge);
		if (dodist)
			imshow("D",dx);
		imshow("T",temp);
		waitKey(2000);
	}
	waitKey();
}
