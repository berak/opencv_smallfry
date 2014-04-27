#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv_db.h"

#include <cstdio>
#include <iostream>
using namespace cv;
using namespace std;

extern Ptr<opencv_db> createSqlite3Db();
extern Ptr<opencv_db> createMysqlDb();
extern Ptr<opencv_db> createMongoDb(int port=27017);
extern Ptr<opencv_db> createFsDb();
extern Ptr<opencv_db> createMemDb();

int main()
{
    Ptr<opencv_db> db = createMongoDb();//10099);
    db->open("moloch","localhost",0,0);
    //db->open("moloch","paulo.mongohq.com",0,0);
    //db->open("moloch","paulo.mongohq.com","paula","17z17z17z");
    //Ptr<opencv_db> db = createSqlite3Db();
    //bool ok = db->open("ocv.sqlite",0,0,0);
    //Ptr<opencv_db> db = createFsDb();
    //bool ok = db->open(0,0,0,0);

    char * table = "img4";
    bool ok = false;
    //ok = db->drop(table);
    //ok = db->create(table);
    Mat m;
    m = imread("../../demo/tuna.jpg",1);
    //ok = db->write(table,"tuna",m);
    
    //m = imread("../../demo/lena.jpg",1);
    //ok = db->write(table,"lena",m);

    vector<int> prm; prm.push_back(IMWRITE_PNG_COMPRESSION); prm.push_back(2);
    vector<uchar> mb; imencode(".png",m,mb,prm);
    ok = db->write(table,"tuna.png",Mat(mb));

    Mat md = Mat::eye(3,4,CV_64F);
    ok = db->write(table,"eye",md);
    
    Mat m2;
    //ok = db->read(table,"tuna_x",m2);
    //if ( ok && m2.rows )
    //    imshow("success!",m2), waitKey();
    //
    //ok = db->read(table,"lena",m2);
    //if ( ok && m2.rows )
    //    imshow("success!",m2), waitKey();
    //
    ok = db->read(table,"tuna.png",m2);
    if ( ok && m2.rows ) {
        m2 = imdecode(m2,1);
        if ( ok && m2.rows )
            imshow("success!",m2), waitKey();
    }
     
 
  return false;
}