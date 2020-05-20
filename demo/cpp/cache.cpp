#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/img_hash.hpp"
#include <iostream>
#include <fstream>
#include <map>
using namespace cv;
using namespace std;

map<int64,string> dict;
typedef map<int64,string> dict_iter;
Ptr<img_hash::PHash> ph;

/*
int cleanup(string c_in) {

    vector<String> fn;
    glob(c_in, fn, true);
    cout << "orig " << fn.size() << endl;
    for(size_t i=0; i<fn.size(); i++)
    {
        Mat im = imread(fn[i]);
        int64 key=hashit(im);
        //cout << key << " " << out << endl;
        auto it = dict.find(key);
        if (it == dict.end()) {
            dict[key] = fn[i];
        } else {
            cerr << fn[i] << " " << it->second << endl;
            //unlink(fn[i]);
            Mat im2 = imread(it->second);
            string del = fn[i];
            dict[key] = it->second;
            if (im.total() * im.elemSize() > im2.total()* im.elemSize()) {
                del = it->second;
                dict[key] = fn[i];
            }
            if ((! im.empty()) && (!im2.empty())) {
                imshow("A",im);
                imshow("B",im2);
                waitKey(50);
            }
            string cmd = "del " + del;
            char d = system(cmd.c_str());
        }
        if (i%999==0)cout << dict.size() << endl;
        //cout << out <<fn[i] << endl;
    }
    FileStorage fs("cache.yml.gz",1);
    fs << "map" << "[";
    for (auto it=dict.begin(); it!=dict.end(); it++) {
        fs << "{" << format("_%08x",it->first) << it->second << "}";
    }
    fs << "]";
    fs.release();

    return 128;
}
*/


int64 hashit(const Mat &im) {
    Mat_<uchar> out;
    ph->compute(im,out);
    int64 key=0;
    for (int j=0; j<out.cols; j++) {
        key |= unsigned(out(j)) << (j*8);
    }
    return key;
}

bool dupli(string fn, const Mat &im) {
    int64 key = hashit(im);
    auto it = dict.find(key);
    if (it == dict.end()) {
        dict[key] = fn;
        return false;
    }
    return true;
}

int db(string c_in) {
    vector<String> fn;
    glob(c_in, fn, true);
    cout << "orig " << fn.size() << endl;
    for(size_t i=0; i<fn.size(); i++)
    {
        Mat im = imread(fn[i]);
        int64 key=hashit(im);
        dict[key] = fn[i];
    }
    return 0;
    ofstream of("cache.txt");
    for (auto e=dict.begin(); e != dict.end(); e++) {
        of << e->first << " " << e->second << endl;
    }
    of.close();
}

int main( int argc, char** argv )
{
    ifstream in("cache.txt");
    int64 k;
    string v;
    while(in >> k >> v) {
        dict[k] = v;
    }
    in.close();
    cout << "dict " << dict.size() << endl;
    ph = img_hash::PHash::create();
    int K=0;
    String c_in("c:\\Users\\ppp\\AppData\\Local\\Mozilla\\Firefox\\Profiles\\p49ig5is.default-release\\cache2\\entries");
    String c_out("c:\\data\\cache\\");
    //return db(c_out);

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
            && (!dupli(fn[i],im))
            && (im.rows >= 128)
            && (im.cols >= 128)
            && imwrite(n, im))
        {
            cerr << "+ " << n << " " << nb <<endl;
            K ++;
        }
        else
            cerr << "- " << n << " " << nb << endl;

    }
    cerr << "saved " << K << " images." << endl;
    ofstream of("cache.txt");
    for (auto e=dict.begin(); e != dict.end(); e++) {
        of << e->first << " " << e->second << endl;
    }
    of.close();
    return 0;
}
