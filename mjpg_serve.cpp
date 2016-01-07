//
// a single-threaded, multi client(using select), debug webserver - streaming out mjpg.
//
//   on win, _WIN32 has to be defined, must link against ws2_32.lib (socks on linux are for free)
//
//   currently, vlc does not like it
//






//
// socket related abstractions:
//
#ifdef _WIN32  
    #include <winsock.h>
    #include <windows.h>
    #include <time.h>
    #define PORT	    unsigned long
    #define ADDRPOINTER   int*
    struct _INIT_W32DATA
    {
	   WSADATA w;
	   _INIT_W32DATA() { WSAStartup( MAKEWORD( 2, 1 ), &w ); }
    } _init_once;
#else		/* ! win32 */
    #include <unistd.h>
    #include <sys/time.h>
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netdb.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #define PORT	    unsigned short
    #define SOCKET	  int
    #define HOSTENT	 struct hostent
    #define SOCKADDR	struct sockaddr
    #define SOCKADDR_IN  struct sockaddr_in
    #define ADDRPOINTER  unsigned int*
    #define INVALID_SOCKET -1
    #define SOCKET_ERROR   -1
#endif /* _WIN32 */



#include <iostream>
using std::cerr;
using std::endl;

#include "opencv2/opencv.hpp"
using namespace cv;



class MJPGWriter
{
    SOCKET sock;
    fd_set master;
    int timeout; // master sock timeout, shutdown after timeout millis.
    int quality; // jpeg compression [1..100]

    int _write( int sock, char *s, int len ) 
    { 
        if ( len < 1 ) { len = strlen(s); }
        return ::send( sock, s, len, 0 );
    }

public:

    MJPGWriter() 
        : sock(INVALID_SOCKET) 
        , timeout(200000)
        , quality(30)
    {
		FD_ZERO( &master );
    }

    MJPGWriter(int port) 
        : sock(INVALID_SOCKET) 
        , timeout(200000)
        , quality(30)
    {
		FD_ZERO( &master );
        open(port);
    }

    ~MJPGWriter() 
    {
        release();
    }

    bool release()
    {
        if ( sock != INVALID_SOCKET )
            ::shutdown( sock, 2 );
        sock = (INVALID_SOCKET);
        return false;
    }

    bool open( int port )
    {
        sock = ::socket (AF_INET, SOCK_STREAM, IPPROTO_TCP) ;

        SOCKADDR_IN address;	   
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_family      = AF_INET;
        address.sin_port        = ::htons(port);
        if ( ::bind( sock, (SOCKADDR*) &address, sizeof(SOCKADDR_IN) ) == SOCKET_ERROR )
        {
            cerr << "error : couldn't bind sock "<<sock<<" to port "<<port<<"!" << endl;
            return release();
        }
        if ( ::listen( sock, 10 ) == SOCKET_ERROR )
        {
            cerr << "error : couldn't listen on sock "<<sock<<" on port "<<port<<" !" << endl;
            return release();
        }
		FD_SET( sock, &master );	
        return true;
    }

    bool isOpened() 
    {
        return sock != INVALID_SOCKET; 
    }

    bool write(const Mat & frame)
    {
        fd_set rread = master;
        struct timeval to = {0,timeout};
        SOCKET maxfd = sock+1;

        if ( ::select( maxfd, &rread, NULL, NULL, &to ) <= 0 )
            return true; // nothing broken, there's just noone listening

        std::vector<uchar>outbuf;
        std::vector<int> params;
        params.push_back(IMWRITE_JPEG_QUALITY);
        params.push_back(quality);
        cv::imencode(".jpg", frame, outbuf, params);
        int outlen = outbuf.size();

        #ifdef _WIN32 
		for ( unsigned i=0; i<rread.fd_count; i++ )
        {
            SOCKET s = rread.fd_array[i];    // fd_set on win is an array, while ...
        #else         
        for ( int s=0; s<maxfd; s++ )
        {
            if ( ! FD_ISSET(s,&rread) )      // ... on linux it's a bitmask ;)
                continue;
        #endif                   
            if ( s == sock ) // request on master socket, accept and send main header.
            {
                int         addrlen = sizeof(SOCKADDR);
                SOCKADDR_IN address = {0};	   
                SOCKET      client  = ::accept( sock,  (SOCKADDR*)&address, &addrlen );
                if ( client == SOCKET_ERROR )
                {
                    cerr << "error : couldn't accept connection on sock " << sock<< " !" << endl;
                    return false;
                }
                maxfd=(maxfd>client?maxfd:client);
                FD_SET( client, &master );
                _write( client,"HTTP/1.0 200 OK\r\n",0);
                _write( client,
	                "Server: Mozarella/2.2\r\n"
	                "Accept-Range: bytes\r\n"
	                "Connection: close\r\n"
                    "Max-Age: 0\r\n"
                    "Expires: 0\r\n"
                    "Cache-Control: no-cache, private\r\n"
                    "Pragma: no-cache\r\n"
	                "Content-Type: multipart/x-mixed-replace; boundary=mjpegstream\r\n"
	                "\r\n",0);
                cerr << "new client " << client << endl;
            } 
            else // existing client, just stream pix
            {
                char head[400];
                sprintf(head,"--mjpegstream\r\nContent-Type: image/jpeg\r\nContent-Length: %lu\r\n\r\n",outlen);
                _write(s,head,0);
                int n = _write(s,(char*)(&outbuf[0]),outlen);
                //cerr << "known client " << s << " " << n << endl;
                if ( n < outlen )
                {
                    cerr << "kill client " << s << endl;
                    ::shutdown(s,2);
                    FD_CLR(s,&master);
                }
            }
        }
        return true;
    }
};



int main()
{
    VideoCapture cap;
    bool ok = cap.open(0);
	if ( ! ok ) 
	{
		printf("no cam found ;(.\n");
		return 1;
	}

    MJPGWriter wri(7777);

    while( cap.isOpened() && wri.isOpened() )
    {
        Mat frame;
        cap >> frame;

        wri.write(frame);

        imshow("lalala",frame);
        int k = waitKey(10); 
        if ( k==27 )
            break;
    }
    return 0;
}


