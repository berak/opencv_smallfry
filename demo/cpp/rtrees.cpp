#include "opencv2/opencv.hpp"
using namespace cv;

#include <fstream>
#include <iostream>
using namespace std;


int main() {
Mat data(5,5,CV_32F);
Mat lbl(5,1, CV_32S);

Ptr<ml::RTrees> tree = ml::RTrees::create();
tree->train(data,0,lbl);
tree->save("rtree.xml");

int _ = system("ls -l *.xml");

string xml;
ifstream is("rtree.xml");
if (is) {
    // get length of file:
    is.seekg (0, is.end);
    int length = is.tellg();
    is.seekg (0, is.beg);

    // ...buffer contains the entire file...
    char * buffer = new char [length];
    is.read (buffer,length);
    is.close();

    xml = string(buffer);
    delete[] buffer;
}
cout << xml.size() <<  " in memory." << endl;
cout << xml << endl;

FileStorage fs(xml, FileStorage::MEMORY|FileStorage::READ);

Ptr<ml::RTrees> another_tree = ml::RTrees::create();

another_tree->read(fs.getFirstTopLevelNode());

cout << another_tree->isTrained() << " " << another_tree->getVarCount() << endl;


}