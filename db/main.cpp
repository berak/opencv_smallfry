#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv_db.h"

#include <cstdio>
#include <iostream>
using namespace cv;
using namespace std;

extern Ptr<opencv_db> createSqlite3Db();
extern Ptr<opencv_db> createMysqlDb();
extern Ptr<opencv_db> createFsDb();

int main()
{
    Ptr<opencv_db> db = createMysqlDb();
    bool ok = db->open("ocv","localhost","root",0);
    //Ptr<opencv_db> db = createSqlite3Db();
    //bool ok = db->open("ocv.sqlite",0,0,0);
    //Ptr<opencv_db> db = createFsDb();
    //bool ok = db->open(0,0,0,0);

    cerr << ok << " open" << endl;
    char * table = "img4";
    
    //// ok = db->exec(format("create table %s (name TEXT, t INTEGER, w INTEGER, h INTEGER, pix BLOB);", table).c_str());
    ok = db->create(table);
    Mat m;
    m = imread("../demo/tuna.jpg",1);
    ok = db->write(table,"tuna",m);

    m = imread("../demo/lena.jpg",1);
    ok = db->write(table,"lena",m);

    vector<int> prm; prm.push_back(IMWRITE_PNG_COMPRESSION); prm.push_back(2);
    vector<uchar> mb; imencode(".png",m,mb,prm);
    ok = db->write(table,"lena.png",Mat(mb));

    Mat md = Mat::eye(3,4,CV_64F);
    ok = db->write(table,"eye",md);
    
    Mat m2;
    ok = db->read(table,"tuna",m2);
    if ( ok && m2.rows )
        imshow("success!",m2), waitKey();
    
    ok = db->read(table,"lena",m2);
    if ( ok && m2.rows )
        imshow("success!",m2), waitKey();
    
    ok = db->read(table,"lena.png",m2);
    if ( ok && m2.rows ) {
        m2 = imdecode(m2,1);
        if ( ok && m2.rows )
            imshow("success!",m2), waitKey();
    }
     
 
  return false;
}