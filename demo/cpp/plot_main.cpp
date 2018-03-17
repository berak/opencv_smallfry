#include <opencv2/opencv.hpp>
#include <opencv2/plot.hpp>
#include <iostream>
#include <cmath>
using namespace cv;
using namespace std;


// this is more a stress-test of the underlying plot library.

void plot_test() {
    Mat draw;
    vector<double> x, y;
    for (int i=0; i<100; i++) {
        x.push_back(i*0.1);
        y.push_back(sin(i*0.1));
    }
    Ptr<plot::Plot2d> plot = plot::createPlot2d(x, y);
    plot->render(draw);

    x.clear(); y.clear();
    for (int i=0; i<100; i++) {
        x.push_back(i*0.1);
        y.push_back(cos(i*0.1));
    }
    plot = plot::createPlot2d(x, y);
    plot->setPlotLineColor(Scalar(0,200,0));
    plot->render(draw);

    imshow("curve", draw);
    waitKey();
}

double gauss(double x, double mu, double sig2) {
    return exp(-(x-mu)*(x-mu)/(2*sig2)) / sqrt(2*CV_PI*sig2);
}
void plot_gauss() {
    Mat draw;
    Ptr<plot::Plot2d> plot;
    vector<double> x, y;
    float B = 5;
    float mu = -2;
    float sig2 = 0.1;
    for (double i=-B; i<B; i+=0.1) {
        x.push_back(i);
        y.push_back(gauss(i,mu,sig2));
    }
    plot = plot::createPlot2d(x, y);
    plot->setMaxY(1.5);
    plot->render(draw);

    sig2 = 5;
    mu = -4;
    x.clear(); y.clear();
    for (double i=-B; i<B; i+=0.1) {
        x.push_back(i);
        y.push_back(gauss(i,mu,sig2));
    }
    plot = plot::createPlot2d(x, y);
    plot->setPlotLineColor(Scalar(0,200,0));
    plot->setMaxY(1.5);
    plot->render(draw);

    imshow("curve", draw);
    waitKey();
}

int main(int argc, char **argv)
{
    plot_gauss();
    return 0;

    CommandLineParser parser(argc,argv,
        "{im || image path }"
        "{W  || width }"
        "{H  || height }"
        "{xm || min X }"
        "{xM || max X }"
        "{ym || min Y }"
        "{yM || max Y }"
        "{sx |1| x scale }"
        "{sy |1| y scale }"
        "{ph |-1| initial phase }"
        "{of |0| y offset }"
        "{dx |0.003| speed }"
        "{help h usage ? || show this message }"
    );
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    vector<double> x, y;
    float sx = parser.get<float>("sx");
    float sy = parser.get<float>("sy");
    float dx = parser.get<float>("dx");
    float off = parser.get<float>("of");
    float phase = parser.get<float>("ph");
    for (int i=0; i<1235; i++) {
        x.push_back(phase*sx);
        y.push_back(off+cos(7*phase)*sin(3*phase)*sy);
        //y.push_back(off+log(phase)*sy);
        phase += dx;
    }
    Ptr<plot::Plot2d> plot = plot::createPlot2d(x, y);

    Mat plotMat = imread(parser.get<String>("im"));// if we have a valid image, use that.
    if (plotMat.empty() || (parser.has("W") && parser.has("H"))) {
        plotMat.create(parser.get<float>("H"), parser.get<float>("W"), CV_8UC3);
        plotMat.setTo(0);
    }
    if (parser.has("xm")) plot->setMinX(parser.get<float>("xm"));
    if (parser.has("xM")) plot->setMaxX(parser.get<float>("xM"));
    if (parser.has("ym")) plot->setMinY(parser.get<float>("ym"));
    if (parser.has("yM")) plot->setMaxY(parser.get<float>("yM"));

    plot->render(plotMat);
    imshow("MTF", plotMat);
    imwrite("./plot.jpg", plotMat);
    waitKey();
    return 0;
}

