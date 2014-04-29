//
//

#include "opencv2/core/core.hpp"
#include "birds/birds.h"
#include "opencv_db.h"

#include <cstdio>
#include <iostream>
using namespace cv;
using namespace std;

//
// table and image names get concatenated to a key: "tab.img"
// each item is a list: type,w,h,pixels(binary) (RPUSH,LRANGE,LINDEX,..)
//
struct RedisDb : opencv_db
{
    int port;
    int sock;

    int readline( char *line ) {
	    int nb = 0;
	    line[0] = 0;
	    while(true) 
	    {
		    int b = Birds::ReadByte(sock);
		    if ( b < 0 )
			    break;
		    if ( b == '\r' )
			    continue;
		    if ( b == '\n' )
			    break;
		    line[nb] = b;
		    nb ++;
	    }
	    line[nb]=0;
	    return nb;
    }
    bool _error( const char *a=0, const char *b=0 )
    {
        if ( a ) cerr << a;
        if ( b ) cerr << " " << b;
        cerr << endl;
        return false;
    }

    bool _result()
    {
        char line[500];
        readline(line);
        if ( line[0]!='+' || line[2]!='K' )
            return _error(line);
        return true;
    }
    bool _handshake(const string &mes)
    {
        int n = Birds::Write(sock, (char*)mes.c_str(), 0);
        if ( n<=0 ) 
            return _error("writing", mes.c_str() );

        return _result();
    }

    RedisDb(int port=6307) : port(port),sock(-1) {}
    ~RedisDb() { close();}

    virtual bool open( const char * db, const char * host, const char * user, const char * pw ) 
    {
    	sock = Birds::Client((char*)host,port);

        if ( sock < 0 )
            return _error("connecting to ", host );

        if ( pw && pw[0] ) 
	    {
            string msg(format("auth %s\r\n", pw ));
            if ( Birds::Write(sock,(char*)msg.c_str(),0)<1 ) 
                return false;
	    }
        if ( ! _result() )
            return false;

        // try to select the db, if it was a number:
        int dbn = -1;
        sscanf(db,"%i",&dbn);
        if ( dbn > -1 )
        {
		    string mes = format(
                "*2\r\n"
                "$6\r\n"
                "SELECT\r\n"
                "$1\r\n"
                "%d\r\n", dbn );

            if ( ! _handshake(mes) ) 
                return false;
        }
        return true; 
    }


    virtual bool write( const std::string & table, const std::string & name, const cv::Mat & mat ) 
    { 
        string key = table + "." + name;
        string t = format("%d",mat.type());
        string w = format("%d",mat.cols);
        string h = format("%d",mat.rows);
        int numbytes=mat.total() * mat.elemSize();
        // my 'precompiled statement' looks pretty much like gnu assembler ;)
        string stmt_pre = format(
            "*6\r\n"
            "$5\r\n"
            "RPUSH\r\n"
            "$%d\r\n" 
            "%s\r\n"
            "$%d\r\n" 
            "%s\r\n"
            "$%d\r\n" 
            "%s\r\n"
            "$%d\r\n" 
            "%s\r\n"
            "$%d\r\n", key.length(),key.c_str(),
                       t.length(),t.c_str(),
                       w.length(),w.c_str(),
                       h.length(),h.c_str(),numbytes);
        Birds::Write(sock,(char*)stmt_pre.c_str(),stmt_pre.length());
        Birds::Write(sock,(char*)mat.data,numbytes);
        Birds::Write(sock,"\r\n",2);

        return _result();
    }

    virtual bool read ( const std::string & table, const std::string & name, cv::Mat & mat ) 
    { 
        string key = table + "." + name;
        string stmt_pre = format(
            "*4\r\n"
            "$6\r\n"
            "LRANGE\r\n"
            "$%d\r\n"
            "%s\r\n"
            "$1\r\n"
            "0\r\n"
            "$1\r\n"
            "3\r\n", key.length(),key.c_str());
        Birds::Write(sock,(char*)stmt_pre.c_str(),stmt_pre.length());

        int t=0,w=0,h=0,nb=0;
        char line[600];
        readline(line);
        if ( line[0]!='*' || line[1]!='4' )
            return _error("key not found : ",key.c_str());
        readline(line);//$1
        readline(line); t=atoi(line);
        readline(line);//$1
        readline(line); w=atoi(line);
        readline(line);//$1
        readline(line); h=atoi(line);
        readline(line); nb=atoi(line+1);
        if ( nb<1 ) 
            return _error("protocol error, expected: $num_bytes : ", line);

        mat.create(h,w,t);
        uchar *d = mat.data;
        for ( int bread=0; bread<nb; bread++ )
        {
            *d++ = Birds::ReadByte(sock); // TODO: read blocks
        }
        readline(line); // flush last readline
        return !mat.empty(); 
    }

    virtual bool close() { 
        return Birds::Close(sock)!=0;
    }
};

Ptr<opencv_db> createRedisDb(int port=6307) { return makePtr<RedisDb>(port); }
