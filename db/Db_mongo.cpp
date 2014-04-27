#include "opencv2/core/core.hpp"
#include "opencv_db.h"

extern "C" {
 #include "mongo.h"
}

#include <cstdio>
#include <iostream>
using namespace cv;
using namespace std;

//
// it connects to the default port(27017)
//
struct MongoDb : opencv_db
{
    mongo_connection conn[1];
    string db;
    int port;

    MongoDb(int port=27017)
        : port(port)
    {
        #ifdef _WIN32
            WSADATA out; 
            int r =  WSAStartup(MAKEWORD(2,2), &out); 
            cerr << r << endl;
        #endif
    }

    bool _error(const char * foo=0,const char * bar=0)
    {
        bson b;
        mongo_cmd_get_last_error(conn,db.c_str(),&b);
        bson_print(&b);
        if ( foo ) cerr << " in " << foo;
        if ( bar ) cerr << " : " << bar;
        cerr << endl;
        return false;
    }


    virtual bool open( const char * db, const char * host, const char * user, const char * pw ) 
    {
        this->db = db;
        // mongo can't resolve hostnames, so i have to.
        string ip = host;
        unsigned long i_addr = ::inet_addr( host );
        if ( i_addr == INADDR_NONE ) 
        {   
            HOSTENT *hostentry  = ::gethostbyname( host );
            if ( hostentry )
                i_addr =  *(unsigned long *)hostentry->h_addr_list[0];

            ip = format("%d.%d.%d.%d", i_addr & 0xFF, (i_addr >> 8) & 0xFF,  (i_addr >> 16) & 0xFF, (i_addr >> 24) & 0xFF);	
        }	
        cerr << ip << endl;

        mongo_connection_options opts = {0};  
        strcpy(opts.host, ip.c_str());
        opts.port = port;

        if ( mongo_connect( conn , &opts ) )
            return _error("failed to connect");

        if ( (!user) || (!pw) )
            return true;
        if ( mongo_cmd_authenticate(conn, db, user, pw) != 0)
            return _error("authenticate");
        return true; 
    }

    virtual bool exec( const char * statement ) 
    { 
        return true; 
    }

    virtual bool create( const std::string & table ) 
    { 
        return true; // we don't have to explicitly create collections
    }
    virtual bool drop  ( const std::string & table ) 
    { 
        bson b;
        return (mongo_cmd_drop_collection( conn, db.c_str(), table.c_str(),&b)==0); 
    } 

    virtual bool write( const std::string & table, const std::string & name, const cv::Mat & mat ) 
    { 
        string ns = db + "." + table;

        bson_buffer bb;
        bson_buffer_init( & bb );

        bson_append_new_oid( &bb, "_id" );
        bson_append_string( &bb , "n" , name.c_str() );
        bson_append_int( &bb , "t" , mat.type() );
        bson_append_int( &bb , "w" , mat.cols );
        bson_append_int( &bb , "h" , mat.rows );
        bson_append_binary( &bb, "pix", bson_bindata, ((const char*)(mat.data)), mat.total()*mat.elemSize() );

        bson b;
        bson_from_buffer(&b, &bb);
        mongo_insert( conn , ns.c_str() , &b );
        bson_destroy(&b);

        return true; // aww, rrly, this can't go wrong ever ?
    }

    virtual bool read ( const std::string & table, const std::string & name, cv::Mat & mat ) 
    { 
        bson_buffer bb;
        bson_buffer_init( & bb );
        bson_append_string( &bb , "n" , name.c_str() );

        bson b;
        bson_from_buffer(&b, &bb);

        string ns = db + "." + table;
        mongo_cursor * cursor = mongo_find( conn , ns.c_str() , &b , 0 , 1 , 0 , 0 );
        mongo_cursor_next(cursor);
        {
            int t=0,w=0,h=0;
            void *d=0;
            bson_iterator it;
            bson_iterator_init(&it, cursor->current.data);
            while(bson_iterator_next(&it))
            {
                const char * k = bson_iterator_key(&it);
                if ( !strcmp("t",k) ) {  t = bson_iterator_int(&it);  }
                if ( !strcmp("w",k) ) {  w = bson_iterator_int(&it);  }
                if ( !strcmp("h",k) ) {  h = bson_iterator_int(&it);  }
                if ( !strcmp("pix",k) ){ d = (void*)bson_iterator_bin_data(&it);  }
            }
            mat = Mat(h,w,t,d).clone();
        }
        mongo_cursor_destroy(cursor);
        bson_destroy(&b);
        return (! mat.empty()); 
    }

    virtual bool close() 
    { 
        mongo_destroy( conn );
        return true;
    }
};


Ptr<opencv_db> createMongoDb(int port=27017) { return makePtr<MongoDb>(port); }
