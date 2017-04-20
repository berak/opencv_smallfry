#include <iostream>
#include <fstream>
#include <string>

using namespace std;

int main(int argc, char **argv) {
	string fn = "eye.xml";
	ifstream in(fn.c_str());
	string all,line;
	while (getline(in,line))
		all += line + "\n";

	ofstream out("eye.h");
	out << "const char xml[] = {";
	for (size_t i=0; i<all.length(); i++) {
		if (i%16==0) out << "\n";
		out << int(all[i]) << ", ";
	}
	out << "10};\n";
	return 0;
}
