#include <string>
#include <iostream>

using namespace std;

template <class T=int> T crc(const string &s) {
    const unsigned n = sizeof(T);
	unsigned r[n] = {0};
    for (size_t i=0; i<s.size(); i++) {
    	r[i%n] ^= unsigned(s[i]);
    }
    T res=0;
    for (int i=0; i<n; i++) {
    	res += (r[i] << (8*i));
    }
    return res;
}

int main(int argc, char **argv) {
	cerr << (argc-1) << " args." << endl;
	for (int i=1; i<argc; i++) {
		cerr << argv[i] << " " << crc<int>(argv[i]) << endl;
	}
    return 0;
}
