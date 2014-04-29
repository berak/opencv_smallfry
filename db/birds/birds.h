#ifndef __birds_onboard__
#define __birds_onboard__



namespace Birds 
{
    int Server(int port);
    int Accept(int serv);
    int Select(int sock, int timeout, int(*cb)(int s,char *mess));

    int Client(char *host, int port);
    int Select(int sock, int timeout);

    int Write(int sock, char *s, int len);

    char ReadByte(int sock);
    char * Read(int sock);
    char * Read( int sock, char *buffer, int & len );

    char * Error();
    
    int Close(int sock);
}



#endif //__birds_onboard__
