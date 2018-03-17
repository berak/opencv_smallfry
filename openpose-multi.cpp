//
//  openpose for multiple persons
//
//  download the caffemodel from: http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
//  also the original(!) pose_deploy_linevec.prototxt
//
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <iostream>
using namespace std;


// see: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
const String PART_NAMES[] {
"Nose","Neck","RShoulder","RElbow","RWrist","LShoulder" ,"LElbow" ,"LWrist","RHip","RKnee","RAnkle","LHip",
"LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "Background"};
// }

// a lookup table in the format: [model_type][body_part_id][from/to]
int POSE_PAIRS[2][17][2] = {
{   // COCO, https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/pose/poseParameters.cpp#L259
    {1,2}, {1,5}, {2,3},
    {3,4}, {5,6}, {6,7},
    {1,8}, {8,9}, {9,10},
    {1,11}, {11,12}, {12,13},
    {1,0}, {0,14},
    {14,16}, {0,15}, {15,17}
},
{   // MPI
    {0,1}, {1,2}, {2,3},
    {3,4}, {1,5}, {5,6},
    {6,7}, {1,14}, {14,8}, {8,9},
    {9,10}, {14,11}, {11,12}, {12,13},
    {-1,-1}, {-1,-1}, {-1,-1}
} };

// COCO
int POSE_PAF[19][2] =
{
    {31,32}, {39,40}, {33,34}, {35,36}, {41,42}, {43,44}, {19,20}, {21,22}, {23,24}, {25,26},
    {27,28}, {29,30}, {47,48}, {49,50}, {53,54}, {51,52}, {55,56}, {37,38}, {45,46}
};


float line_integral(const Mat &x, Point a, Point b)
{
    LineIterator it(x,a,b);
    float val=0;
    while(it.pos() != b)
    {
        val += *(float*)(*it);
        it ++;
    }
    return (val / it.count);
}

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv,
        "{ help           | false | print this help message }"
        "{ proto          | c:/data/mdl/pose_deploy_linevec.prototxt | model configuration }"
        "{ model          | c:/data/mdl/pose_iter_440000.caffemodel | model weights }"
        "{ image          | c:/p/ocv/demo/pers.jpg| path to image file (containing a single person) }"
        "{ threshold      |0.1| threshold value for the heatmap}"
    );

    String modelTxt = parser.get<string>("proto");
    String modelBin = parser.get<string>("model");
    String imageFile = parser.get<String>("image");
    float thresh = parser.get<float>("threshold");
    if (parser.get<bool>("help") || modelTxt.empty() || modelBin.empty() || imageFile.empty())
    {
        cout << "A sample app to demonstrate human pose detection with a pretrained OpenPose dnn." << endl;
        parser.printMessage();
        return 0;
    }

    // fixed input size for the pretrained network
    int W_in = 368;
    int H_in = 368;

    Net net = readNetFromCaffe(modelTxt, modelBin);

    Mat img = imread(imageFile);
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << imageFile << std::endl;
        exit(-1);
    }

    Mat inputBlob = blobFromImage(img, 1.0 / 255, Size(W_in, H_in), Scalar(0, 0, 0), false, false);
    net.setInput(inputBlob);
    Mat result = net.forward();
/*

    MatShape ms1 { inputBlob.size[0], inputBlob.size[1] , inputBlob.size[2], inputBlob.size[3] };
    size_t nlayers = net.getLayerNames().size() + 1;        // one off for the hidden input layer
    for (size_t i=0; i<nlayers; i++) {
        Ptr<Layer> lyr = net.getLayer((unsigned)i);
        vector<MatShape> in,out;
        net.getLayerShapes(ms1,i,in,out);
        cout << format("%-38s %-13s ", (i==0?"data":lyr->name.c_str()), (i==0?"Input":lyr->type.c_str()));
        for (auto j:in)  cout << "i" << Mat(j).t() << "  "; // input(s) size
        for (auto j:out) cout << "o" << Mat(j).t() << "  "; // output(s) size
        for (auto b:lyr->blobs) {                           // what the net trains on, e.g. weights and bias
            cout << "b[" << b.size[0];
            for (size_t d=1; d<b.dims; d++) cout << ", " << b.size[d];
            cout << "]  ";
        }
        cout << endl;
    }

    cout << "flops " << net.getFLOPS(ms1) << endl;
    vector<double> timings;
    cout << "ticks " << net.getPerfProfile(timings) << endl;
    cout << "time " << Mat(timings).t() << endl;
*/

    int pidx, npairs;
    int nparts = result.size[1];
    int H = result.size[2];
    int W = result.size[3];
    if (nparts == 19 || nparts == 57)
    { // COCO
        pidx   = 0;
        npairs = 17;
    }
    else if (nparts == 16)
    { // MPI
        pidx   = 1;
        npairs = 14;
    }
    else
    {
        cerr << "there should be 19/57 body parts for the COCO model or 16 for the MPI one, but this model has " << nparts << " parts." << endl;
        return (0);
    }
    nparts = min(nparts,18); // we don't care for background and the PAF's (here)
    vector<vector<Point>> points(19); // ((x y conf) per limb) per person)

    Mat stripx(H,0,CV_32F,0.0f);
    Mat stripy(H,0,CV_32F,0.0f);
    Mat stripp(H,0,CV_32F,0.0f);

    for (int n=0; n<nparts; n++)
    {
        // Slice heatmap of corresponging body's part.
        Mat hm(H, W, CV_32F, result.ptr(0,n));
        Mat heatMap = hm.clone();
        hconcat(stripp,hm,stripp);
        // the poor man's nms. paint the maximum black, and try again.
        Point pm(-1,-1);
        double conf;
        for (int p=0; p<10; p++)
        {
            minMaxLoc(heatMap, 0, &conf, 0, &pm);
            circle(heatMap, pm, 2, Scalar(0),-1);
            if (conf<thresh) break;
            cout << n << " " << p << " " << pm << " " << conf << endl;
            points[n].push_back(pm);
        }
        cout << n << " " << PART_NAMES[n] << " " << points[n].size() << " points" << endl;
    }

    vector<pair<Point,Point>> limbs;
    for (size_t n=0; n<nparts; n++)
    {
        Mat x(H, W, CV_32F, result.ptr(0,POSE_PAF[n][0]));
        Mat y(H, W, CV_32F, result.ptr(0,POSE_PAF[n][1]));
        hconcat(stripx,x,stripx);
        hconcat(stripy,y,stripy);

        int pid1 = POSE_PAIRS[pidx][n][0];
        int pid2 = POSE_PAIRS[pidx][n][1];
        cout << "pid " << n << " " << pid1 << " " << pid2 << endl;

        for (size_t i=0; i<points[pid1].size(); i++)
        {
            Point2f a = points[pid1][i], best;

            float bestval=-1;
            for (size_t j=0; j<points[pid2].size(); j++)
            {
                Point2f b = points[pid2][j];
                Point2f dir = (b - a) / norm(b - a);
                float dx = line_integral(x,a,b);
                float dy = line_integral(y,a,b);
                //Point2f P(dx, dy);
                //Point2f dir_aP = (P) / norm(P);
                //normalize(P,dir_aP);
                //float val = abs(dir_ab.x*dir_aP.x + dir_ab.y * dir_aP.y);
                float val = abs(dx*dir.x + dy*dir.y);
                cout << n << " " << i << " " << j << " " << a << " " << b << " " << val << endl;
                if (val > bestval)
                {
                    bestval = val;
                    best = b;
                }
            }
            if (bestval > thresh)
            {
                limbs.push_back(make_pair(a,best));
                cout << "select " << PART_NAMES[pid1] << " " << pid1 << " " <<  a << " " << best << " " << bestval << endl;
            }
            else
            {
                cout << "drop   " << PART_NAMES[pid1] << " " <<  pid1 << " " <<  a << " " << best << " " << bestval << endl;
            }
        }
    }

    float SX = float(img.cols) / W;
    float SY = float(img.rows) / H;
    cout << limbs.size() << " limbs" << endl;
    for (size_t i=0; i<limbs.size(); i++)
    {
        Point a = limbs[i].first;
        Point b = limbs[i].second;
        line(img, Point(a.x*SX,a.y*SY), Point(b.x*SX,b.y*SY), Scalar(0,200,0), 2);
        circle(img, Point(a.x*SX,a.y*SY), 3, Scalar(0,0,200), -1);
        circle(img, Point(b.x*SX,b.y*SY), 3, Scalar(0,0,200), -1);
    }
    imshow("p",stripp);
    imshow("x",stripx*3);
    imshow("y",stripy*3);
    imshow("OpenPose using OpenCV", img);
    waitKey();
    return 0;
}
