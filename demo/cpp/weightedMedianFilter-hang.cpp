/*
 * weightedMedianFilter-hang.cc
 *
 *  Created on: Apr 14, 2018
 *      Author: amyznikov
 */

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>


using namespace std;
using namespace cv;
using namespace cv::ximgproc;

int main(int argc, char *argv[])
{
  string input_filename = "ss-2.003.png", output_filename = "wmf.png";
  Mat input_image, output_image;

  int input_depth;

  int raduis = 5;
  double sigma = 1.0;
  int weight_type = WMF_OFF;

  for ( int i = 1; i < argc; ++i ) {

    if ( strcmp(argv[i], "--help") == 0 ) {
      printf("usage:\n");
      printf(" weightedMedianFilter-hang input-image.png -o output-image.png \n"
          "     [-r radius]\n"
          "     [-s sigma]\n"
          "\n");
      return 0;
    }

    if ( strcmp(argv[i], "-r") == 0 ) {
      if ( ++i >= argc ) {
        fprintf(stderr, "Missing argument after %s\n", argv[i]);
        return 1;
      }
      if ( sscanf(argv[i], "%d", &raduis) != 1 ) {
        fprintf(stderr, "invalid parameter value : %s\n", argv[i]);
        return 1;
      }
    }
    else if ( strcmp(argv[i], "-s") == 0 ) {
      if ( ++i >= argc ) {
        fprintf(stderr, "Missing argument after %s\n", argv[i]);
        return 1;
      }
      if ( sscanf(argv[i], "%lf", &sigma) != 1 ) {
        fprintf(stderr, "invalid parameter value : %s\n", argv[i]);
        return 1;
      }
    }
    else if ( strcmp(argv[i], "-o") == 0 ) {
      if ( ++i >= argc ) {
        fprintf(stderr, "Missing argument after %s\n", argv[i]);
        return 1;
      }
      output_filename = argv[i];
    }
    else if ( input_filename.empty() ) {
      input_filename = argv[i];
    }
    else {
      fprintf(stderr, "invalid argument %s\n", argv[i]);
      return 1;
    }
  }


  if ( input_filename.empty() ) {
    fprintf(stderr, "No input image specified\n");
    return 1;
  }

  input_image = cv::imread(input_filename, IMREAD_UNCHANGED);
  if ( !input_image.data ) {
    fprintf(stderr, "imread('%s') fails\n", input_filename.c_str());
    return 1;
  }

  fprintf(stderr, "input: %dx%d depth=%d channels=%d type=%d\n",
      input_image.cols, input_image.rows,
      input_image.depth(),
      input_image.channels(),
      input_image.type());


  if ( (input_depth = input_image.depth()) != CV_8U && input_depth != CV_32F )  {
    input_image.convertTo(input_image, CV_32F);
  }

  weightedMedianFilter(Mat::ones(input_image.size(), CV_8U),
      input_image, output_image, raduis, sigma,  weight_type);


  fprintf(stderr, "output: %dx%d depth=%d channels=%d \n",
      output_image.cols, output_image.rows,
      output_image.depth(), output_image.channels());

  if ( output_image.depth() != input_depth ) {
    output_image.convertTo(output_image, input_depth);
  }

  if ( !imwrite(output_filename, output_image) ) {
    fprintf(stderr, "imwrite('%s') fails\n", output_filename.c_str());
    return 1;
  }

  return 0;
}
