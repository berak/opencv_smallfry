
#include "opencv2/core/core.hpp"
#include "opencv_db.h"

#include "pg/include/libpq-fe.h"

#include <cstdio>
#include <iostream>
using namespace cv;
using namespace std;

struct PostgresDb : opencv_db
{
	  
	PGconn     *conn;

    bool  _error( const char * fn, const char * txt=0 )
	{
        if ( fn  ) cerr << fn  << " "; 
        if ( txt ) cerr << txt << " ";
		cerr << PQerrorMessage(conn) << endl; 
        close();
		return 0; 
	}

public:
    PostgresDb() 
        : conn(0)
	{}

	~PostgresDb() 
	{
		close();
	}


	virtual bool open( const char * db, const char * host, const char * user, const char * pw ) 
	{
		string conninfo;
		if ( host )
            conninfo += format("host=%s ",host );
		if ( user )
            conninfo += format("user=%s ",user );
		if ( pw )
            conninfo += format("password=%s ",pw );
		if ( db )
            conninfo += format("dbname=%s",db );

        conn = PQconnectdb(conninfo.c_str());
		if (conn == 0)
			return _error(0,conninfo.c_str());
		return 1;
	}


	virtual bool exec( const char * statement ) 
	{
        if (conn == 0)
			return _error("no connection",statement);

		PGresult *res = PQexec(conn, statement);
		if (res == 0)
			return _error("exec",statement);
	    PQclear(res);
		return 1;
	}

	virtual bool create( const char * table ) 
	{
        return exec(format("create table %s ( name TEXT UNIQUE, t INTEGER, w INTEGER, h INTEGER, pix BYTEA);",table).c_str());
    }

	virtual bool drop( const char * table ) 
	{
        return exec(format("drop table %s;",table).c_str());
    }

	virtual bool close() 
	{
		if ( conn )
			PQfinish(conn);
		conn = 0;
		return 1;
	}

    virtual bool write( const char * table, const char * name, const cv::Mat & mat ) 
    { 
        if (conn == 0)
			return _error("no connection",0);

        char command[312];
        sprintf(command, "insert into %s values('%s',%d,%d,%d,$1::bytea);", table, name, mat.type(), mat.rows, mat.cols);

        size_t ilen(mat.total() * mat.elemSize()), olen(0);
        char * data = (char*)PQescapeBytea(mat.data, ilen, &olen); 

        const char *paramValues[1] = { data };
        const int  paramLengths[1] = { olen };
        const int  paramFormats[1] = { 1 };
        PGresult * res = PQexecParams(conn, command, 1, NULL, paramValues, paramLengths, NULL, 1);
        int r = PQresultStatus(res);
		if ( r<1 || r>3 )
            cerr << PQerrorMessage(conn) << endl; 
        PQclear(res);
        PQfreemem(data);
        return r>0 && r<4;
    }

    virtual bool read ( const char * table, const char * name, cv::Mat & mat ) 
    { 
        if (conn == 0)
			return _error("no connection",0);
        PGresult *res = PQexec(conn, format("select * from %s where name='%s';",table,name).c_str());
        //PGresult *res = PQexec(conn, format("select name,t,w,h, decode(pix,'hex') from %s where name='%s';",table,name).c_str());
        int r = PQresultStatus(res);
        if ( r>0&&r<4)
        {
            int t = atoi(PQgetvalue(res, 0, 1));
            int w = atoi(PQgetvalue(res, 0, 2));            
            int h = atoi(PQgetvalue(res, 0, 3));
            size_t to_length=0;
            uchar * data = PQunescapeBytea((unsigned char *)PQgetvalue(res, 0, 4), &to_length);
            // baaah, it's still bloody octal encoded. giving up here. ;(
            Mat m(h,w,t,data);
            mat = m.clone();
            PQfreemem(data);
        } else  cerr << PQerrorMessage(conn) << endl; 
        return !mat.empty(); 
    }

};

Ptr<opencv_db> createPostgresDb() { return makePtr<PostgresDb>(); }
