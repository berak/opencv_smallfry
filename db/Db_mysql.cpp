#include "opencv2/core/core.hpp"
#include "opencv_db.h"

#include <windows.h> // TODO: bad enough that SOCKET is missing from mysql.h, but do i need all of windows.h?
extern "C" {
 #include "mysql.h"
}

#include <cstdio>
#include <iostream>
using namespace cv;
using namespace std;

//
// we're only using stuff from the base mysql distro here, no further adapters required
// names are restricted to 64 chars
// the blobsize is restricted to 4mb
// prepared statements look noisy, but allow more control,
//   so we can avoid escaped text and have a real binary storage instead.
// it connects to the default port(3306)
//
struct MysqlDb : opencv_db
{
    MYSQL *mysql;

    MysqlDb() 
        : mysql(mysql_init(NULL)) 
    {}

    bool _error(const char * foo=0,const char * bar=0)
    {
        const char * err = mysql_error(mysql);
        if ( err ) cerr << err;
        if ( foo ) cerr << " in " << foo;
        if ( bar ) cerr << " : " << bar;
        cerr << endl;
        return false;
    }

    virtual bool open( const char * db, const char * host, const char * user, const char * pw ) 
    {
        if ( ! mysql ) 
            return _error("open");

        if ( mysql_real_connect(mysql, host, user, pw, db, 0, NULL, 0) == NULL ) 
            return _error("connect");

        return true; 
    }

    virtual bool exec( const char * statement ) 
    { 
        if (mysql_query(mysql, statement))
            return _error("exec");

        return true; 
    }

    virtual bool create( const std::string & table ) 
    { 
        return exec(format("create table %s (name VARCHAR(64) UNIQUE, t INTEGER, w INTEGER, h INTEGER, pix MEDIUMBLOB);", table.c_str()).c_str()); 
    }
    virtual bool drop  ( const std::string & table ) 
    { 
        return exec(format("drop table %s;",table.c_str()).c_str()); 
    } 

    static int _stmt_close(MYSQL_STMT *s) {  mysql_stmt_close(s); return 0; }
    virtual bool write( const std::string & table, const std::string & name, const cv::Mat & mat ) 
    { 
        int t = mat.type();
        int w = mat.cols;
        int h = mat.rows;
        string q = format("insert into %s values(?, ?, ?, ?, ?);", table.c_str());

        MYSQL_STMT * stmt = mysql_stmt_init(mysql);
        raii<MYSQL_STMT> ras(stmt,_stmt_close);

        if ( ! stmt )
            return _error("write init statement");
        
        if ( mysql_stmt_prepare(stmt,q.c_str(),q.length()) ) 
            return _error("write prepare statement"); 

        MYSQL_BIND bind[5] = {0};
        bind[0].buffer_type= MYSQL_TYPE_STRING;
        bind[0].buffer= (void*)name.c_str();
        bind[0].buffer_length= name.length();

        bind[1].buffer_type= MYSQL_TYPE_LONG;
        bind[1].buffer= (char *)(&t);

        bind[2].buffer_type= MYSQL_TYPE_LONG;
        bind[2].buffer= (char *)(&w);

        bind[3].buffer_type= MYSQL_TYPE_LONG;
        bind[3].buffer= (char *)(&h);

        bind[4].buffer_type= MYSQL_TYPE_MEDIUM_BLOB;
        bind[4].buffer= (char *)(mat.data);
        bind[4].buffer_length= mat.total()*mat.elemSize();

        if ( mysql_stmt_bind_param(stmt,bind) ) 
            return _error("write bind params"); 
        
        if ( mysql_stmt_execute(stmt) ) 
            return _error("write execute"); 

        return true; 
    }

    virtual bool read ( const std::string & table, const std::string & name, cv::Mat & mat ) 
    { 
        MYSQL_STMT * stmt = mysql_stmt_init(mysql);
        raii<MYSQL_STMT> ras(stmt,_stmt_close);
        
        // no need to read the name again.
        string q = format("select t,w,h,pix from %s where name = '%s';",table.c_str(),name.c_str());
        if ( ! stmt )
            return _error("read init statement",q.c_str());

        if ( mysql_stmt_prepare(stmt,q.c_str(),q.length()) ) 
            return _error("read prepare statement",q.c_str()); 

        if ( mysql_stmt_execute(stmt) )
            return _error("read mysql_stmt_execute",q.c_str());

        vector<char> d(4*1024*1204); // TODO: 4mb enough?
        int t=0,w=0,h=0;
        MYSQL_BIND bind[4] = {0};

        bind[0].buffer_type= MYSQL_TYPE_LONG;
        bind[0].buffer= (void*)(&t);

        bind[1].buffer_type= MYSQL_TYPE_LONG;
        bind[1].buffer= (void*)(&w);

        bind[2].buffer_type= MYSQL_TYPE_LONG;
        bind[2].buffer= (void*)(&h);

        bind[3].buffer_type= MYSQL_TYPE_BLOB;
        bind[3].buffer= (void*)(&d[0]);
        bind[3].buffer_length= d.size();

        if ( mysql_stmt_bind_result(stmt,bind) ) 
            return _error("read bind result",q.c_str()); 

        if ( mysql_stmt_store_result(stmt) )
            return _error("read store result",q.c_str());

        if ( mysql_stmt_fetch(stmt) )
            return _error("read fetch result",q.c_str());

        //cerr << w << " " << " " << h << " " << t << endl;
        mat = Mat(h,w,t,((void*)(&d[0]))).clone();
        return (! mat.empty()); 
    }

    virtual bool close() 
    { 
        mysql_close(mysql);
        return true;
    }
};


Ptr<opencv_db> createMysqlDb() { return makePtr<MysqlDb>(); }
