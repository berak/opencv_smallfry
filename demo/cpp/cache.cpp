#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
   int k=0;
    String c_in("c:\\Users\\ppp\\AppData\\Local\\Mozilla\\Firefox\\Profiles\\p49ig5is.default-release\\cache2\\entries");
    String c_out("c:\\data\\cache\\");
    if (argc>1)
    {
        c_out += String(argv[1]);
        String cmd = String("@mkdir ") + c_out;
        system(cmd.c_str());
        c_out += "\\";
    }
    vector<String> fn;
    glob(c_in, fn, true);
    for(size_t i=0; i<fn.size(); i++)
    {
        ifstream in(fn[i], ios_base::binary);
        in.seekg(0, in.end);
        size_t nb = in.tellg();
        in.close();
        //cout << nb << " " ;
        if (nb > 1e7) { // 10mb
            cout << "bummer : " << nb << " " << fn[i] << endl;
            continue;
        }
        String v2 = fn[i].substr(c_in.size()+1);
        String n = c_out + v2 + ".png";
        Mat im = imread(fn[i]);

        if ((! im.empty())
            && (im.rows >= 128)
            && (im.cols >= 128)
            && imwrite(n, im))
        {
            cerr << "+ " << n << " " << nb <<endl;
            k ++;
        }
        else
            cerr << "- " << n << " " << nb << endl;

    }
    cerr << "saved " << k << " images." << endl;
    return 0;
}
