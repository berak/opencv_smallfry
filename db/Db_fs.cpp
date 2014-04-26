#include "opencv2/core/core.hpp"
#include "opencv_db.h"

#include <cstdio>
#include <iostream>
using namespace cv;
using namespace std;

//
// use opencv'S FileStorage / xml
//

struct FsDb : opencv_db
{
    FileStorage fs;
    const string ending;
    FsDb(const char *e=".xml") : ending(e) {}

    virtual bool open( const char * db, const char * host, const char * user, const char * pw ) { return true; }
    virtual bool exec( const char * statement ) { return false; }
    virtual bool create( const std::string & table ) { return false; }
    virtual bool drop  ( const std::string & table ) { return false; } 

    virtual bool write( const std::string & table, const std::string & name, const cv::Mat & mat ) 
    { 
        if ( ! fs.isOpened() )
        { 
            FileStorage fs2(table+ending,FileStorage::READ); 
            if (fs2.isOpened())
            {
                fs2.release();
                fs.open(table+ending,FileStorage::APPEND);
            }
            else
            {
                fs.open(table+ending,FileStorage::WRITE);
            }
        }
        if ( ! fs.isOpened() ) return false;
        fs << name << mat ;
        fs.release();
        return true; 
    }

    virtual bool read ( const std::string & table, const std::string & name, cv::Mat & mat ) 
    { 
        if ( ! fs.isOpened() ) fs.open(table+ending,FileStorage::READ);
        if ( ! fs.isOpened() ) return false;
        fs [ name ] >> mat ;
        return !mat.empty(); 
    }

    virtual bool close() { 
        fs.release();
        return true;
    }
};


Ptr<opencv_db> createFsDb() { return makePtr<FsDb>(); }
