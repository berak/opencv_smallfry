#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include <fstream>

using namespace cv;
using namespace std;

//typedef map<string, vector<int> > Hier;
typedef map<int, vector<int> > Hier;

int main(int argc, char** argv)
{
	ifstream tr1("c:/p/darknet/data/9k.tree");
	ifstream nms("c:/p/darknet/data/9k.names");
	ifstream lbl("c:/p/darknet/data/9k.labels");
	vector<string> names;
	map<string,int> ids;
	map<string,int> parents;
	Hier hier;
	while(nms.good()) {
		string k;
		getline(nms,k);
		if (k.empty()) break;
		names.push_back(k);
	}
	while(lbl.good()) {
		string k;
		lbl >> k;
		ids[k] = ids.size();
	}
	while(tr1.good()) {
		string k;
		tr1 >> k;
		int p;
		tr1 >> p;
		hier[p].push_back(ids[k]);
	}
	cout << "//" << hier.size() << endl;
	cout << "//" <<ids.size() << endl;
	cout << "//" << names.size() << endl;
	int N=1000;
	/*auto rec = [&](int id, int indent){

		if (hier.find(id)!= hier.end()) {
			while(indent--) cout << " ";
		}
	};*/
	cout << "digraph imagenet {";
 	for (Hier::iterator it=hier.begin(); it!= hier.end(); it++) {
 		//cout << it->first << " " << (it->first>=0 ? names[it->first] : "_") << ": ";
 		/*cout << (it->first>=0 ? names[it->first] : "_") << endl;
 		for (size_t i=0; i<it->second.size(); i++) {
 			cout << "  " << names[it->second[i]];
 		}
 		cout << endl;
 		if (N--<0) break;
 		*/
 		for (size_t i=0; i<it->second.size(); i++) {
 			cout << "\"" << (it->first>=0 ? names[it->first] : "imagenet") << "\"";
 			cout << " -> \"" << names[it->second[i]] << "\"" << endl;
 		}
 	}
 	cout << "}" << endl;
    return 0;
}
