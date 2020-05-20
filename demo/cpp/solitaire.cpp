#include <vector>
#include <iostream>
using std::cin;
using std::cout;
using std::endl;


const static int ncolors = 4;
const static int nvals = 13;
const static int ntotal = ncolors * nvals;
const char vals[nvals] = {'2','3','4','5','6','7','8','9','0','B','D','K','A'};
const char cols[ncolors] = {'+','-','+','-'};
const char mpty = ' ';
static int v2id(char v) {
	return v=='2' ? 0 :
		   v=='3' ? 1 :
		   v=='4' ? 2 :
		   v=='5' ? 3 :
		   v=='6' ? 4 :
		   v=='7' ? 5 :
		   v=='8' ? 6 :
		   v=='9' ? 7 :
		   v=='0' ? 8 :
		   v=='B' ? 9 :
		   v=='D' ? 10 :
		   v=='K' ? 11 : 12;
		   //v=='A' ? 12;
}
struct card {
	int val,col;
	card(int v=-1, int c=-1) : val(v), col(c) {}
	void render() const {
		cout << (col<0?mpty:cols[col]) << (val<0?mpty:vals[val]) << " ";
	}
};
typedef std::vector<card> stack;
card pop(stack &s) {
	if (!s.empty()) {
		stack::iterator last = s.begin() + (s.size()-1);
		card c = *last;
		s.erase(last);
		return c;
	}
	return card {-1,-1};
}
void render_item(const stack &s, int i) {
	card c;
	if (!s.empty()) {
		if (i == -1) {
			c = s.back();
		} else if (i < int(s.size())) {
			c = s[i];
		}
	}
	c.render();
}
void render_last(const stack &s) {
	render_item(s,-1);
	//card c = (!s.empty()) ? s.back() : card();
	//c.render();
}
stack init_pool() {
	stack pool;
	for (int c=0; c<ncolors; c++)
	for (int v=0; v<nvals; v++) {
		pool.push_back(card{v,c});
	}
    for (int t=0; t<ntotal; t++) {
        card &tmp = pool[t];
        int r = rand() % ntotal;
        pool[t] = pool[r];
        pool[r] = tmp;
    }
	return pool;
}
struct game {
	const static int cstep = 3;
	const static int naces = 4;
	const static int ncols = 6;
	const static int nrows = 10;
	stack cols[ncols], aces[naces], pool, closed;
	game() { init(); }
	void init() {
		closed.clear();
		pool = init_pool();
		for (auto a:aces) a.clear();
		for (auto c:cols) c.clear();
		for (int i=0; i<ncols; i++) {
			for (int j=0; j<i+1; j++) {
				cols[i].push_back(pop(pool));
			}
		}
	}
    int newCard() {
    	for (int i=0; i<cstep-1; i++) {

    	}
    }
	bool move(int from, int to) {
		return false;
	}
	void render() const {
		render_last(pool);
		for (int a=0; a<naces; a++) {
			render_last(aces[a]);
		}
		cout << endl;
		for (int r=0; r<nrows; r++) {
			for (int c=0; c<ncols; c++) {
				render_item(cols[c],r);
			}
			cout << endl;
		}
	}
};
int main() {
	game g;
	for (;;) {
		g.render();
	    char r,c;
	    cin >> r;
	    if (r=='c') {
	    	continue;
	    }

	    cin >> c;
	 }
    return 0;
}