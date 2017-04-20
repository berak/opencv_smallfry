#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <math.h>

#define M_PI    (4*atan(1)) // since M_PI is not defined by c++11

static void meshgrid(const cv::Mat &xgv, const cv::Mat &ygv,
              cv::Mat1i &X, cv::Mat1i &Y)
{
    cv::repeat(xgv.reshape(1,1), ygv.total(), 1, X);
    cv::repeat(ygv.reshape(1,1).t(), 1, xgv.total(), Y);
}

static void meshgridTest(const cv::Range &xgv, const cv::Range &ygv,
                         cv::Mat1i &X, cv::Mat1i &Y)
{
    std::vector<int> t_x, t_y;
    for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
    for (int i = ygv.start; i <= ygv.end; i++) t_y.push_back(i);
    meshgrid(cv::Mat(t_x), cv::Mat(t_y), X, Y);
}

bool equirectangularToStereographic(cv::Mat &inImage_org,cv::Mat &result,
    double CAMDIST = 8.,double WORLDROTATION = 0.) {
    CV_Assert(CAMDIST > 0.);

    cv::Mat inImage;
    inImage_org.convertTo(inImage, CV_32FC3, 1.0/255.);

    double w = inImage.cols;
    double h = inImage.rows;
    double z = w/CAMDIST;

    std::cout << "image w:" << w << " image h:" << h << " camera z:" << z << std::endl;

    double rads = 2*M_PI/w;

    auto d = [](float i, float j) -> float {
        double ret = i - j / 2;
        return ret;
    };

    auto r = [d,w,h](double x,double y) -> double {
        double x2 = d(x,w);
        double y2 = d(y,h);
        double ret = sqrt(x2 * x2 + y2 * y2);
        return ret;
    };

    auto rho = [z,r](double x,double y) -> double {
        double ret = r(x,y) / z;
        return ret;
    };

    auto theta = [rho](cv::Mat &x,cv::Mat &y,cv::Mat &ret){
        ret = x.clone();
        cv::MatIterator_<double> itx = x.begin<double>();
        cv::MatIterator_<double> ity = y.begin<double>();
        cv::MatIterator_<double> itret = ret.begin<double>();
        for(;itx != x.end<double>(); ++itx,++ity,++itret) {
            *itret = 2. * atan(rho(*itx,*ity));
        }
    };

    auto a = [d,h,w](cv::Mat &x,cv::Mat &y,cv::Mat &ret){
        ret = x.clone();
        cv::MatIterator_<double> itx = x.begin<double>();
        cv::MatIterator_<double> ity = y.begin<double>();
        cv::MatIterator_<double> itret = ret.begin<double>();
        for(;itx != x.end<double>(); ++itx,++ity,++itret) {
            *itret = atan2(d(*ity,h),d(*itx,w))  - M_PI / 4.;
        }
    };

    cv::Mat1i X, Y;
    meshgridTest(cv::Range(1,(int)w), cv::Range(1, (int)h), X, Y);

    cv::Mat pixX,pixY;
    X.convertTo(pixX, CV_64FC1, 1.0);
    Y.convertTo(pixY, CV_64FC1, 1.0);

    cv::Mat lat,lon;
    theta(pixX,pixY,lat);
    a(pixX,pixY,lon);

    auto lat_mod = [](cv::Mat &lat,cv::Mat &ret){
        ret = lat.clone();
        cv::MatIterator_<double> it = lat.begin<double>();
        cv::MatIterator_<double> itret = ret.begin<double>();
        for(;it != lat.end<double>(); ++it,++itret) {
            *itret = fmod(*it + M_PI + M_PI,M_PI) - (M_PI / 2.);
        }
    };

    auto lon_mod = [&WORLDROTATION](cv::Mat &lon,cv::Mat &ret){
        ret = lon.clone();
        cv::MatIterator_<double> it = lon.begin<double>();
        cv::MatIterator_<double> itret = ret.begin<double>();
        for(;it != lon.end<double>(); ++it,++itret) {
            *itret = fmod(*it + M_PI + M_PI*2. + WORLDROTATION,M_PI*2.) - M_PI;
        }
    };

    lat_mod(lat,lat);
    lon_mod(lon,lon);
    cv::Mat xe,ye;
    xe = -(-lon/rads) + w/2.0;
    ye = -(lat/rads) + h/2.0;
    cv::Mat outImage = cv::Mat::zeros(inImage.size(),inImage.type());
    cv::Mat xef,yef;
    xe.convertTo(xef, CV_32FC1, 1.0);
    ye.convertTo(yef, CV_32FC1, 1.0);
    std::vector<cv::Mat> planes;
    cv::split(inImage_org, planes);
    cv::Mat dummy_plane = cv::Mat::zeros(planes[0].size(),planes[0].type()) + 255;
    planes.push_back(dummy_plane);
    cv::Mat merged;
    cv::merge(planes,merged);
    cv::remap(merged,outImage,xef,yef,cv::INTER_LINEAR);
    cv::split(outImage, planes);
    cv::Mat mask = (planes[3]).clone();
    cv::bitwise_not (mask,mask);
    planes.pop_back();
    cv::merge(planes,outImage);
    cv::inpaint(outImage, mask, result, 3, cv::INPAINT_NS);


    return true;
}

const cv::String keys =
	"{help h usage ? |          | show help            }"
	"{@image         |          | input image file     }"
	"{@result        |result.tif| output image file    }"
	"{F flip         |        0 | flip horizontally    }"
	"{D dist         |        8 | camera distance      }"
	"{W w            |        0 | world rotation       }"
	"{S show         |          | should show result   }"
	;

int main(int argc, char *argv[]) {
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("The little planets.");

    bool should_flip = parser.has("F") ? true: false;
    bool should_show = parser.has("S") ? true: false;
    double world_rotation = parser.get<double>("W");
    double camdist = parser.get<double>("D");
    cv::String imagefile = parser.get<cv::String>("@image");
    cv::String resultfile = parser.get<cv::String>("@result");

    if (parser.has("h") || !parser.check() || imagefile.empty()) {
        parser.printMessage();
        return -1;
    }

    CV_Assert(camdist >= 1.0);

    cv::Mat image = cv::imread(imagefile);
    if(image.empty()) {
        std::cout << "Error: image file " << imagefile << " read failed." << std::endl;
        return -1;
    }

    if(should_flip) cv::flip(image, image, 0);

    cv::Mat result;

    equirectangularToStereographic(image,result,camdist,world_rotation);

    cv::imwrite(resultfile,result);

    if(should_show) {
        cv::namedWindow("result",cv::WINDOW_NORMAL);
        cv::imshow("result",result);
        cv::waitKey(0);
    }

    return 0;
}
