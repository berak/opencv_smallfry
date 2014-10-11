#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv_db.h"

#include <cstdio>
#include <iostream>
using namespace cv;
using namespace std;

extern Ptr<opencv_db> createSqlite3Db();
extern Ptr<opencv_db> createMysqlDb();
extern Ptr<opencv_db> createMongoDb(int port=27017);
extern Ptr<opencv_db> createRedisDb(int port=6379);
extern Ptr<opencv_db> createFsDb();
extern Ptr<opencv_db> createMemDb();
extern Ptr<opencv_db> createPostgresDb();

int main()
{
    Ptr<opencv_db> db = createPostgresDb();//10099);
    bool ok = db->open("p4p4","localhost","p4p4","p4p4p4p4");
    //Ptr<opencv_db> db = createRedisDb(6379);//10099);
    //bool ok = db->open("0","gateway-1.simpleredis.com","paula","ib58a1c19e9c3b4c6c6c775d1d9887f9fb52fb57cz");
    //Ptr<opencv_db> db = createSqlite3Db();
    //bool ok = db->open("ocv.sqlite",0,0,0);
    //Ptr<opencv_db> db = createMemDb();
    //bool ok = db->open(0,0,0,0);

    cerr << "1 " << ok << endl;
    if ( ! ok ) return 1;
    char * table = "img4";
    ok = db->drop(table);
    cerr << "drop  " << table << " " << ok << endl;
    ok = db->create(table);
    cerr << "creat " << table << " " << ok << endl;

    Mat md = Mat::eye(3,4,CV_64F)*127;
    ok = db->write(table,"eye",md);
    cerr << "2 " << ok << endl;
    
    Mat m2;
    ok = db->read(table,"eye",m2);
    cerr << "3 " << ok << endl;

    if ( ok && m2.rows )
        imshow("success!",m2), waitKey();
    
    Mat m = imread("../../demo/tuna.jpg",1);
    vector<int> prm; prm.push_back(IMWRITE_PNG_COMPRESSION); prm.push_back(2);
    vector<uchar> mb; imencode(".png",m,mb,prm);
    ok = db->write(table,"tuna.png",Mat(mb));
    cerr << "4 " << ok << endl;

    Mat m5;
    ok = db->read(table,"tuna.png",m5);
    cerr << "5 " << ok << endl;

    if ( ok && m5.rows ) {
        m2 = imdecode(m5,1);
        if ( ok && m2.rows )
            imshow("success!",m2), waitKey();
    }
     
 
  return false;
}
