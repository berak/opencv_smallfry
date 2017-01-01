
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>

#include <iostream>

using namespace cv;
using namespace std;

#ifdef _WIN32
 const char SEP = '\\';
#else
 const char SEP = '/';
#endif

//
// imgfolder
//  + pers1
//   + img1
//   + img2
//   + img3
//  + pers2
//   + img1
//   + img2
//   + img3
// ...
//
// you can pass like 'images/*.png', too!
//
int readdir(String dirpath, std::vector<std::string> &names, std::vector<int> &labels, size_t maxim, int minp=10, int maxp=10)
{

    int r0 = dirpath.find_last_of(SEP)+1;

    vector<String> vec;
    glob(dirpath,vec,true);
    if ( vec.empty())
        return 0;
    std::vector<std::string> tnames;
    std::vector<int> tlabels;
    int nimgs=0;
    int label=-1;
    String last_n="";
    for (size_t i=0; i<vec.size(); i++)
    {
        // extract name from filepath:
        String v = vec[i];
        String v1 = v.substr(r0);
        int r1 = v1.find_last_of(SEP);
        String n = v1.substr(0,r1);
        if (n != last_n)
        {
            if (nimgs < minp) // roll back
            {
                tlabels.clear();
                tnames.clear();
                if (label >= 0) label --;
            }
            else
            {
                labels.insert(labels.end(),tlabels.begin(),(maxp==-1) ? tlabels.end() : tlabels.begin()+std::min(maxp,(int)tlabels.size()));
                names.insert(names.end(),tnames.begin(),(maxp==-1) ? tnames.end() : tnames.begin()+std::min(maxp,(int)tlabels.size()));
                tnames.clear();
                tlabels.clear();
            }
            nimgs = 0;
            last_n = n;
            label ++;
            if (labels.size() >= maxim) break;
        }
        tnames.push_back(v);
        tlabels.push_back(label);
        nimgs ++;
    }

    return label;
}

int main(int argc, char** argv) {
	String path_caltech="C:/data/caltech/101_ObjectCategories";
    vector<String> vec;
    glob(path_caltech,vec,true);

    for (size_t j=0; j<; i++)
    for (size_t i=0; i<40; i++)
    BACKGROUND_Google


    return 0;
}
