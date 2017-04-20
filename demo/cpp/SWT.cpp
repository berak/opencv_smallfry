#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
using namespace cv;
using namespace std;

struct ValuePoint
{
    int x;
    int y;
    float value;

    // Default constructor.
    ValuePoint(int x = 0, int y = 0) : x(x), y(y), value(0)
    {
    }

    // Copy constructor.
    ValuePoint(const ValuePoint& vp) : x(vp.x), y(vp.y), value(vp.value)
    {
    }

    // Comparator.
    bool operator<(const ValuePoint& vp) const
    {
        return value < vp.value;
    }
};


struct Ray
{
    ValuePoint p;
    ValuePoint q;
    std::vector<ValuePoint> points;
};

// Stroke width transform (SWT).
cv::Mat SWT(const cv::Mat& image, bool darkOnLight, float precision)
{
    // Convert the original image to a new grey scale image.
    cv::Mat gray(image.size(), CV_8U);
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
#ifdef NSTU_DEBUG
    //cv::imshow("Gray", gray);
#endif

    // Detect images's edges.
    cv::Mat edges(image.size(), CV_8U);
    cv::Canny(gray, edges, 150., 400.); //175., 320.
#ifdef NSTU_DEBUG
    cv::imshow("Edges", edges);
#endif

    // Smooth the grey scale image.
    cv::Mat smoothed = cv::Mat(image.size(), CV_32F);
    gray.convertTo(smoothed, CV_32F, 1. / 255.);
    cv::GaussianBlur(smoothed, smoothed, cv::Size(5, 5), 0);
#ifdef NSTU_DEBUG
    //cv::imshow("Smoothed", smoothed);
#endif

    // Create Y and Y gradients.
    cv::Mat gradientX(image.size(), CV_32F, cv::Scalar(1.));
    cv::Mat gradientY(image.size(), CV_32F, cv::Scalar(1.));
    cv::Sobel(smoothed, gradientX, CV_32F, 1, 0);
    cv::Sobel(smoothed, gradientY, CV_32F, 0, 1);
    //cv::Scharr(smoothed, gradientX, CV_32F, 1, 0);
    //cv::Scharr(smoothed, gradientY, CV_32F, 0, 1);
#ifdef NSTU_DEBUG
    //cv::imshow("Gradient X", gradientX);
    //cv::imshow("Gradient Y", gradientY);
#endif

    // Initialize SWT output image, each pixel set to infinity by default.
    cv::Mat swt(image.size(), CV_32F, cv::Scalar(-1.));

    // Collection of detected rays.
    std::vector<Ray> rays;

    // Loop across each row.
    for (int row = 0; row < image.rows; ++row)
    {
        // Loop across each column.
        for (int col = 0; col < image.cols; ++col)
        {
            // Check whether the pixel is an edge.
            if (edges.at<uchar>(row, col) == 0xFFu)
            {
                // The ray to be computed for this pixel.
                Ray ray;

                // Set the first point of the ray.
                ValuePoint p(col, row);
                ray.p = p;
                ray.points.push_back(p);

                // The coordinates of the current pixel on the ray.
                int currPixelX = col;
                int currPixelY = row;

                // The coordinates of the current real point on the ray.
                float currValueX = (float)col + .5f;
                float currValueY = (float)row + .5f;

                // Get x and y components of the gradient.
                float dx = gradientX.at<float>(row, col);
                float dy = gradientY.at<float>(row, col);

                // Get gradient magnitude.
                float mag = sqrt(pow(dx, 2) + pow(dy, 2));

                // Normalize gradient components according to the darkOnLight setting.
                dx = (darkOnLight ? -dx : dx) / mag;
                dy = (darkOnLight ? -dy : dy) / mag;

                // Loop until another edge is found.
                while (true)
                {
                    // Advance in gradient direction by a small amount.
                    currValueX += dx * precision;
                    currValueY += dy * precision;

                    // Proceed only if we moved enough from the last pixel on the ray.
                    if ((int)(floor(currValueX)) != currPixelX || (int)(floor(currValueY)) != currPixelY)
                    {
                        // Convert the current point to the closest pixel.
                        currPixelX = (int)(floor(currValueX));
                        currPixelY = (int)(floor(currValueY));

                        // Terminate if the pixel is outside of the image's boundary.
                        if (currPixelX < 0 || (currPixelX >= swt.cols) || currPixelY < 0 || (currPixelY >= swt.rows))
                            break;

                        // Insert the new point in the ray.
                        ValuePoint pNew(currPixelX, currPixelY);
                        ray.points.push_back(pNew);

                        // Check whether last pixel on gradient direction was an edge.
                        if (edges.at<uchar>(currPixelY, currPixelX) > 0)
                        {
                            // Set the last point of the ray.
                            ray.q = pNew;

                            // Get gradient of last point in the ray.
                            float dxNew = gradientX.at<float>(currPixelY, currPixelX);
                            float dyNew = gradientY.at<float>(currPixelY, currPixelX);
                            mag = sqrt(pow(dxNew, 2) + pow(dyNew, 2));

                            // Normalize gradient components according to the darkOnLight setting.
                            dx = (darkOnLight ? -dx : dx) / mag;
                            dy = (darkOnLight ? -dy : dy) / mag;

                            // Add ray only if its angle with new point's gradient is less than 90�.
                            if (acos(dx * -dxNew + dy * -dyNew) < CV_PI / 2.0)
                            {
                                // Calculate length of ray.
                                float length = sqrt(pow((float)ray.q.x - (float)ray.p.x, 2) + pow((float)ray.q.y - (float)ray.p.y, 2));

                                // Set the value of each pixel covered bay the ray to its length.
                                for (auto& point : ray.points)
                                    swt.at<float>(point.y, point.x) = swt.at<float>(point.y, point.x) < 0 ? length : std::min(length, swt.at<float>(point.y, point.x));

                                // Add this ray to the collection.
                                rays.push_back(ray);
                            }

                            // Exit loop for this pixel.
                            break;
                        }
                    }
                }
            }
        }
    }

    /* Median filter */

    // Loop across all the found rays.
    for (auto& ray : rays)
    {
        // Loop across all points of the ray and set their values.
        for (auto& point : ray.points)
            point.value = swt.at<float>(point.y, point.x);

        // Sort the points by value.
        std::sort(ray.points.begin(), ray.points.end());

        // Find median value in the sorted list of points.
        float median = (ray.points[ray.points.size() / 2]).value;

        // Set new value in the image for each point in this ray.
        for (auto& point : ray.points)
            swt.at<float>(point.y, point.x) = std::min(point.value, median);
    }

    /* Normalization */

    // Get the maximal and minimal values in the SWT image.
    float maxVal = 0;
    float minVal = std::numeric_limits<float>::infinity();
    for (int row = 0; row < image.rows; ++row)
    {
        for (int col = 0; col < image.cols; ++col)
        {
            if (swt.at<float>(row, col) >= 0.f)
            {
                maxVal = std::max(swt.at<float>(row, col), maxVal);
                minVal = std::min(swt.at<float>(row, col), minVal);
            }
        }
    }

    // Get the difference between maximal and minimal values.
    float difference = maxVal - minVal;

    // Normalize each pixel.
    for (int row = 0; row < image.rows; ++row)
        for (int col = 0; col < image.cols; ++col)
            if (swt.at<float>(row, col) < 0)
                swt.at<float>(row, col) = 1;
            else
                swt.at<float>(row, col) = (swt.at<float>(row, col) - minVal) / difference;

    // Convert SWT image from 32 bit float to 8 bit grey scale.
    swt.convertTo(swt, CV_8U, 255.);

    return swt;
}

// Associate pixels to form connected components.

int main(int argc, char **argv)
{
    Mat img = imread("oude-man.jpg", 1);
    Mat result = SWT(img, false, .3);
    imshow("man",result);
    waitKey();
}
