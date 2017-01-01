#include <iostream>
using namespace std;
#include <windows.h>

#if (defined(_M_ARM) || defined(_M_X64) || defined(WINRT)) && _WIN32_WINNT >= 0x501
 const char *s="arm";
#else
 const char *s="bein";
#endif


int getNumberOfCPUs(void)
{
#if defined WIN32 || defined _WIN32
    SYSTEM_INFO sysinfo;
#if (defined(_M_ARM) || defined(_M_X64) || defined(WINRT)) && _WIN32_WINNT >= 0x501
    GetNativeSystemInfo( &sysinfo );
#else
    GetSystemInfo( &sysinfo );
#endif

    return (int)sysinfo.dwNumberOfProcessors;
#elif defined ANDROID
    static int ncpus = getNumberOfCPUsImpl();
    return ncpus;
#elif defined __linux__
    return (int)sysconf( _SC_NPROCESSORS_ONLN );
#elif defined __APPLE__
    int numCPU=0;
    int mib[4];
    size_t len = sizeof(numCPU);

    /* set the mib for hw.ncpu */
    mib[0] = CTL_HW;
    mib[1] = HW_AVAILCPU;  // alternatively, try HW_NCPU;

    /* get the number of CPUs from the system */
    sysctl(mib, 2, &numCPU, &len, NULL, 0);

    if( numCPU < 1 )
    {
        mib[1] = HW_NCPU;
        sysctl( mib, 2, &numCPU, &len, NULL, 0 );

        if( numCPU < 1 )
            numCPU = 1;
    }

    return (int)numCPU;
#else
    return 1;
#endif
}



int main(int argc, char* argv[]) {
  cerr << __cplusplus << endl;
  cerr << s << endl;
  cerr << getNumberOfCPUs() << endl;
#if defined(WIN32)
  cerr << "WIN32" << endl;
#else
  cerr << "no WIN32" << endl;
#endif

#if defined(_WIN32)
  cerr << "_WIN32" << endl;
#else
  cerr << "no _WIN32" << endl;
#endif
  return 0;
}
