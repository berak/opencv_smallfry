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
    bool _error_close( sqlite3_stmt* stmt, const char * fn, const char * txt="" )
    {
        sqlite3_finalize(stmt);
        return _error(fn,txt);
    }

public:
    Sqlite3Db() 
        : sqlite(0)
    {
    }

    ~Sqlite3Db() 
    {
        close();
    }

    virtual bool open( const char * db, const char * host, const char * user, const char * pw ) 
    {
        if ( this->sqlite ) 
            return _error("open", "close running connection first");

        if( sqlite3_open( db, &this->sqlite ) )
            return _error("open");
        return true; 
    }

    virtual bool close() 
    {
        if ( ! this->sqlite ) 
            return false;
        int r = sqlite3_close(this->sqlite);
        this->sqlite = 0;
        if ( r )
            return _error("close");
        return true;
    }


    virtual bool exec( const char * statement ) 
    {
        if ( ! this->sqlite ) 
            return _error("exec","DB_CLOSED");
        sqlite3_stmt *stmt;
        if( sqlite3_prepare_v2(this->sqlite,statement,strlen(statement)+1, &stmt,0) )
            return _error("exec.prepare");
        if( sqlite3_step(stmt) )
            return _error_close(stmt,"exec.step");
        sqlite3_finalize(stmt);
        return true;
    }



    virtual bool write( const char * table, const char * name, const cv::Mat & m ) 
    { 
        sqlite3_stmt *stmt;
        string q = format ("insert into %s values(?,?,?,?,?);", table);
        if( sqlite3_prepare_v2(this->sqlite,q.c_str(),q.length()+1, &stmt, 0) )
            return _error("write.prepare");
        if( sqlite3_bind_text(stmt,1,name,strlen(name),SQLITE_STATIC) )
            return _error_close(stmt,"write.bind");
        if( sqlite3_bind_int(stmt,2,m.type()) )
            return _error_close(stmt,"write.bind");
        if( sqlite3_bind_int(stmt,3,m.rows) )
            return _error_close(stmt,"write.bind");
        if( sqlite3_bind_int(stmt,4,m.cols) )
            return _error_close(stmt,"write.bind");
        if( sqlite3_bind_blob(stmt,5,m.data,m.total()*m.channels(),SQLITE_STATIC) )
            return _error_close(stmt,"write.bind");
        if( sqlite3_step(stmt) )
            return _error_close(stmt,"write.bind");
        sqlite3_finalize(stmt);
        return true; 
    }
 
    // wtf, indexing starts with 1 for bind* calls, but 0 for columns ?
    virtual bool read ( const char * table, const char * name, cv::Mat & mat ) 
    { 
        sqlite3_stmt *stmt;
        string q = format ("select t,w,h,pix from %s where name = ?", table);
        if( sqlite3_prepare_v2(this->sqlite, q.c_str(),-1, &stmt, 0) )
            return _error("read.prepare");
        if( sqlite3_bind_text(stmt, 1, name, -1, SQLITE_STATIC) )
            return _error_close(stmt,"read.bind");
        if ( sqlite3_step(stmt) != SQLITE_ROW )
            return _error_close(stmt,"read.step");
        int t = sqlite3_column_int(stmt,0);
        int h = sqlite3_column_int(stmt,1);
        int w = sqlite3_column_int(stmt,2);
        void *p = const_cast<void*>(sqlite3_column_blob(stmt,3));
        mat = Mat(h,w,t,p).clone();
        sqlite3_finalize(stmt);
        return true; 
    }

    virtual bool create( const char * table ) 
    { 
        std::string q = cv::format("create table %s (name TEXT UNIQUE, t INTEGER, w INTEGER, h INTEGER, pix BLOB);", table);
        return exec(q.c_str());
    }
    virtual bool drop( const char * table ) 
    { 
        std::string q = cv::format("drop table %s;", table);
        return exec(q.c_str());
    }
};

Ptr<opencv_db> createSqlite3Db() { return cv::makePtr<Sqlite3Db>(); }
