//
// the whole purpose of this module is  to serve reference implentations for bechmarks / tests
// no, you probably *don't* want to use any  code below in the real world.
//

#include "opencv2/core/core.hpp"
#include "opencv_db.h"

#include <cstdio>
#include <iostream>
using namespace cv;
using namespace std;

//
//    0.00.  0000.  0.00.  00.    000.
//   0.0.0. 0.     0.0.0. 0. 0.  0.  0.
//   0.  0. 0.000. 0.  0. 0.  0. 0000.
//   0.  0. 0.     0.  0. 0.  0. 0.  0.
//   0.  0. 00000. 0.  0. 0000.  0000.
//
// plain in-memory store, this even ignores tables !
//
#include <map>
struct MemDb : opencv_db
{
    map<string,Mat> mem;

    MemDb() {}

    virtual bool write( const char * table, const char * name, const cv::Mat & mat ) 
    { 
        mem.insert(make_pair(name,mat));
        return true;
    }

    virtual bool read ( const char * table, const char * name, cv::Mat & mat ) 
    { 
        mat= mem[name];
        return !mat.empty(); 
    }

    virtual bool close() { 
        mem.clear();
        return true;
    }
};

Ptr<opencv_db> createMemDb() { return makePtr<MemDb>(); }







//
//  #####  ####  ##      #####  ####  ######   ##    ###     ##     ###   #####
// ##            ##     ##     ##  ##   ##    ####  ## ##   ####   ## ## ##
// ######   ##   ##     ######  ##      ##   ##  ## ##  ## ##  ## ##     ######
// ##       ##   ##     ##       ###    ##   ##  ## #####  ###### #####  ##
// ##      ####  ###### ###### #####    ##    ####  ## ##  ##  ##  ####  ######
//

// use opencv'S FileStorage / xml
// db is ignored, each table is a flat file(containing possibly multiple Mat's).
// read & write are lazy.
//
struct FsDb : opencv_db
{
    FileStorage fs;
    const string ending;
    FsDb(const char *e=".xml") : ending(e) {}

    virtual bool write( const char * table, const char * name, const cv::Mat & mat ) 
    { 
        if ( ! fs.isOpened() )
        { 
            FileStorage fs2(table+ending,FileStorage::READ); 
            if (fs2.isOpened())
            {
                fs2.release();
                fs.open(table+ending,FileStorage::APPEND); // TODO: this will make duplicates.
            }
            else
            {
                fs.open(table+ending,FileStorage::WRITE);
            }
            if ( ! fs.isOpened() ) return false;
        }
        fs << name << mat ;
        fs.release();
        return true; 
    }

    virtual bool read ( const char * table, const char * name, cv::Mat & mat ) 
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
