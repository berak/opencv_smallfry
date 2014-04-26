#ifndef __opencv_db_onboard__
#define __opencv_db_onboard__

#include "opencv2/core/core.hpp"

struct opencv_db 
{
		//! host,user,pw will be 0 in case of local connection(sqlite,slimxml)
        virtual bool open( const char * db, const char * host, const char * user, const char * pw ) = 0;

        virtual bool exec( const char * statement ) = 0;

        virtual bool create( const std::string & table ) = 0;
        virtual bool drop  ( const std::string & table ) = 0; 

        virtual bool write( const std::string & table, const std::string & name, const cv::Mat & mat ) = 0;
        virtual bool read ( const std::string & table, const std::string & name, cv::Mat & mat ) = 0;

        virtual bool close() = 0;

 
        virtual ~opencv_db() {}
};



template <class T>
struct raii {
    typedef int (*dtor)(T*t);
    T *p;
    dtor del;

    raii( T*p=0, dtor del=0 ) : p(p),del(del) { }
    ~raii() { if (p && del) del(p); p=0; }
};


#endif // __opencv_db_onboard__

