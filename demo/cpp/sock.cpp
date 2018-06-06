#include <WinSock2.h>
#include <iostream>
#include <Ws2tcpip.h>
#include <opencv2/opencv.hpp>


using namespace std;


int main()
{
    int r;
    WSAData wsaData;
    WORD DLLVSERION;
    DLLVSERION = MAKEWORD(2, 1);//Winsocket-DLL 版本

                                //用 WSAStartup 開始 Winsocket-DLL
    r = WSAStartup(DLLVSERION, &wsaData);

    //宣告 socket 位址資訊(不同的通訊,有不同的位址資訊,所以會有不同的資料結構存放這些位址資訊)
    SOCKADDR_IN addr;
    int addrlen = sizeof(addr);

    //建立 socket
    SOCKET sListen; //listening for an incoming connection
    SOCKET sConnect; //operating if a connection was found

    sConnect = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    cout << "sConnect is " << sConnect << endl;

    //設定位址資訊的資料
    //inet_ntoa(AF_INET, "172.0.0.1", (int)&addr.sin_addr);
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(1234);

    //設定 Listen
    sListen = socket(AF_INET, SOCK_STREAM, 0);
    bind(sListen, (SOCKADDR*)&addr, sizeof(addr));
    listen(sListen, SOMAXCONN);//SOMAXCONN: listening without any limit

                               //等待連線
    SOCKADDR_IN clinetAddr;



    cout << "waiting.............." << endl;
    sConnect = accept(sListen, (SOCKADDR*)&clinetAddr, &addrlen);
    //sTemp = sConnect;     //saving the descripter of client's socket descripter

    //char *sendbuf;
    char line[100]= "1234567890123456789012345678901234567890";
    send(sConnect, line, (int)strlen(line), 0);
    cout << "sListen is " << sListen << endl;
    cout << "sConnect is " << sConnect << endl;
    cout << "failed" << endl;
    getchar();
}
