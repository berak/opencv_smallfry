#include "opencv2/core/core.hpp"
#include "opencv_db.h"
#include "sqlite/sqlite3.h"

#include <cstdio>
#include <iostream>
using namespace cv;
using namespace std;




struct Sqlite3Db : opencv_db
{
private:


    sqlite3 * sqlite;

    bool _error( const char * fn, const char * txt="" )
    {
        const char * err = sqlite3_errmsg( this->sqlite );
        if ( !strcmp(err,"not an error") )
            return true;
        cerr << format( "%-18s(%s) ( %s )", fn, txt, err ) << endl;
        return false; 
    }
    #define RROR(r) if ( (r) != SQLITE_OK ) { return _error(__FUNCTION__); }
    #define RAII(r) raii<sqlite3_stmt> ras(r,sqlite3_finalize);

public:
    Sqlite3Db() 
        : sqlite(0)
    {
        // printf( "<sqlite %s>\n", sqlite3_libversion() );
    }

    ~Sqlite3Db() 
    {
        close();
    }

    virtual bool open( const char * db, const char * host, const char * user, const char * pw ) 
    {
        if ( this->sqlite ) 
            return _error(__FUNCTION__, "close running connection first");

        RROR( sqlite3_open( db, &this->sqlite ) );
        return true; 
    }

    virtual bool close() 
    {
        if ( ! this->sqlite ) 
            return false;
        RROR( sqlite3_close(this->sqlite) );
        this->sqlite = 0;
        return true;
    }


    virtual bool exec( const char * statement ) 
    {
        if ( ! this->sqlite ) 
            return _error(__FUNCTION__,"DB_CLOSED");

        sqlite3_stmt *stmt;
        RROR( sqlite3_prepare_v2(this->sqlite,statement,strlen(statement)+1, &stmt,0) );
        RAII( stmt );
        RROR( sqlite3_step(stmt) );
        return true;
    }



    virtual bool write( const std::string & table, const std::string & name, const cv::Mat & m ) 
    { 
        sqlite3_stmt *stmt;
        string q = format ("insert into %s values(?,?,?,?,?);", table.c_str());
        RROR( sqlite3_prepare_v2(this->sqlite,q.c_str(),q.length()+1, &stmt, 0) );
        RAII( stmt );
        RROR( sqlite3_bind_text(stmt,1,name.c_str(),name.length(),SQLITE_STATIC) );
        RROR( sqlite3_bind_int(stmt,2,m.type()) );
        RROR( sqlite3_bind_int(stmt,3,m.rows) );
        RROR( sqlite3_bind_int(stmt,4,m.cols) );
        RROR( sqlite3_bind_blob(stmt,5,m.data,m.total()*m.channels(),SQLITE_STATIC) );
        RROR( sqlite3_step(stmt) );
        return true; 
    }
 
    // wtf, indexing starts with 1 for bind* calls, but 0 for columns ?
    virtual bool read ( const std::string & table, const std::string & name, cv::Mat & mat ) 
    { 
        sqlite3_stmt *stmt;
        string q = format ("select t,w,h,pix from %s where name = ?", table.c_str());
        RROR( sqlite3_prepare_v2(this->sqlite, q.c_str(),-1, &stmt, 0) );
        RAII( stmt );
        RROR( sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC) );
        if ( sqlite3_step(stmt) == SQLITE_ROW )
        {
            int t = sqlite3_column_int(stmt,0);
            int h = sqlite3_column_int(stmt,1);
            int w = sqlite3_column_int(stmt,2);
            void *p = const_cast<void*>(sqlite3_column_blob(stmt,3));
            mat = Mat(h,w,t,p).clone();
            return true; 
        }
        return false;
    }

    virtual bool create( const std::string & table ) 
    { 
        std::string q = cv::format("create table %s (name TEXT UNIQUE, t INTEGER, w INTEGER, h INTEGER, pix BLOB);", table.c_str());
        return exec(q.c_str());
    }
    virtual bool drop( const std::string & table ) 
    { 
        std::string q = cv::format("drop table %s;", table.c_str());
        return exec(q.c_str());
    }

#undef RROR
#undef RAII
};

Ptr<opencv_db> createSqlite3Db() { return cv::makePtr<Sqlite3Db>(); }
