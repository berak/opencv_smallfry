#include <opencv2/opencv.hpp>
#include <iostream>
#include <deque>
#include <unordered_map>

using namespace cv;
using namespace std;

struct var {
   enum { NUM,RCT,STR,IO };
   _InputOutputArray io;
   String s;
   Rect r;
   float n;
   var() : n(0) {}
   var(const _InputOutputArray &a) : io(a), n(0) {}
   var(const String &s) : s(s), n(0) {}
   var(const Rect &r) : r(r), n(0) {}
   var(float f) : n(f) {}
   var(const var &v) : io(v.io), s(v.s), r(v.r), n(v.n) {}
   String str() const {
      if (!io.empty())
         return format("[%x %d (%dx%d)]", io.kind(),io.type(),io.size().width,io.size().height);
      if (!s.empty())
         return s;
      if (!r.empty())
         return format("[%d %d %d %d]", r.x,r.y,r.width,r.height);
      return format("%3.3f",n);
   }
   int type() const {
      if (!io.empty()) return IO;
      if (!s.empty()) return STR;
      if (!r.empty()) return RCT;
      return NUM;
   }
};

//! the global state table
typedef unordered_map<string, var> VARS;

// vector<string> helpers
string tostr(const vector<string> &vs) {
   string s("");
   for (auto t: vs) s = (s.empty() ? t : s + " " + t);
   return s;
}
bool has(const vector<string> &z, string s) {
   for (auto p: z)
      if (p==s) return true;
   return false;
}

struct pub_sub {
   virtual void process(VARS &vars) = 0;
   virtual string name() const = 0;
   virtual ~pub_sub() {}

   vector<string> pub, sub;

   bool subscribes(const pub_sub *b) const {
      for (auto p: b->pub)
         if (has(this->sub, p)) return true; // i subscribe to one of b's topics
      return false;
   }
   string str() {
      return name() + "(" + tostr(sub) + "|" + tostr(pub) + ")";
   }
};

struct print_ : pub_sub {
   print_(string n) { sub.push_back(n); }
   virtual void process(VARS &vars) {
      const var &p = vars[sub[0]];
      cerr << sub[0] << " " << p.type() << " " << p.str() << endl;
      if (p.type()==var::IO && max(p.io.size().width, p.io.size().height) < 20)
         cerr << p.io.getMat() << endl;
   }
   virtual string name() const { return "print"; }
};


struct iread : pub_sub {
   Mat m;
   iread(string i, string f, string o) {
      sub.push_back(i);
      sub.push_back(f);
      pub.push_back(o);
   }
   virtual void process(VARS &vars) {
      if (m.empty()) m=imread(vars[sub[0]].s, (int)(vars[sub[1]].n));
      vars[pub[0]] = var(m);
   }
   virtual string name() const { return "read"; }
};

struct iwrite : pub_sub {
   iwrite(string fn, string m) {
      sub.push_back(fn);
      sub.push_back(m);
   }
   virtual void process(VARS &vars) {
      Mat m = vars[sub[1]].io.getMat();
      imwrite(vars[sub[0]].s, m);
   }
   virtual string name() const { return "write"; }
};

struct inv : pub_sub {
   Mat m;
   inv(string s, string d) {
      sub.push_back(s);
      pub.push_back(d);
   }
   virtual void process(VARS &vars) {
      if (vars[sub[0]].type()==var::IO) {
         Mat a = vars[sub[0]].io.getMat();
         bitwise_not(a,m);
         vars[pub[0]] = var(m);
      }
   }
   virtual string name() const { return "inv"; }
};

struct add_ : pub_sub {
   Mat m;
   add_(string s1, string s2, string d) {
      sub.push_back(s1);
      sub.push_back(s2);
      pub.push_back(d);
   }
   virtual void process(VARS &vars) {
      var &v1 = vars[sub[0]];
      var &v2 = vars[sub[1]];
      if (v1.type()==var::IO && v2.type()==var::IO) {
         Mat a = v1.io.getMat();
         Mat b = v2.io.getMat();
         m = a + b;
         vars[pub[0]] = var(m);
      }
   }
   virtual string name() const { return "add"; }
};

struct sub_ : pub_sub {
   Mat m;
   sub_(string s1, string s2, string d) {
      sub.push_back(s1);
      sub.push_back(s2);
      pub.push_back(d);
   }
   virtual void process(VARS &vars) {
      var &v1 = vars[sub[0]];
      var &v2 = vars[sub[1]];
      if (v1.type()==var::IO && v2.type()==var::IO) {
         Mat a = v1.io.getMat();
         Mat b = v2.io.getMat();
         m = a - b;
         vars[pub[0]] = var(m);
      }
   }
   virtual string name() const { return "subtr"; }
};


void assign(string &a, string &b) { b = a; }
void assign(string &a, float &b)  { b = atof(a.c_str()); }
void assign(string &a, Mat &b)    {
   int h,w,t;
   if (3 != sscanf(a.c_str(),"%d %d %d",&h,&w,&t)) return;
   b = Mat(h,w,t);
}
const static char* _n[] = {"num","str","mat"};

template<class T, int N>
struct ref_ : pub_sub {
   T t;
   ref_(string s1, string s2) {
      assign(s1, t);
      pub.push_back(s2);
   }
   virtual void process(VARS &vars) {
      vars[pub[0]] = var(t);
   }
   virtual string name() const { return _n[N]; }
};

typedef ref_<float,0>  num_;
typedef ref_<string,1> str_;
typedef ref_<Mat,2>    mat_;

typedef Ptr<pub_sub> PP;

// basic housekeeping
void insert(deque<PP> &prc, PP x) {
   if (x->pub.empty()) { // noone needs you
      prc.push_back(x);
      return;
   }
   if (x->sub.empty()) { // you need noone (else)
      prc.push_front(x);
      return;
   }
   for (auto q = prc.begin(); q != prc.end(); q++) {
      if ((*q)->subscribes(x)) { // i need you
         prc.insert(q,x); // before q
         return;
      }
   }
   prc.push_back(x);
}
// todo: still need 2 passes,
//  (only relying on one way subscription makes a weak criterion)
void sort(deque<PP> &prc) {
   deque<PP> p2;
   for (auto q: prc) {
      insert(p2, q);
   }
   prc = p2;
}
// check for unresolved subscriptions
bool resolve(deque<PP> &prc, string tok) {
   for (auto p: prc) {
      if (has(p->pub, tok)) // target found
         return true;
   }
   return false;
}
// everyone connected ?
// todo: it returns the opposite..
int sanity(deque<PP> &prc) {
   int fail=0;
   for (auto p: prc) {
      // try to resolve all p's subscriptions
      for (auto s: p->sub) {
         bool ok = resolve(prc, s);
         if (ok) continue;
         cerr << "sanity: unresolved " << s << " in " << p->str() << endl;
         fail ++;
      }
   }
   return fail;
}

int main(int argc, char **argv)
{
   VARS u;
   deque<Ptr<pub_sub>> prc;
   prc.push_back(new print_("im5"));
   prc.push_back(new print_("im3"));
   prc.push_back(new sub_("im2", "im1", "im3"));
   prc.push_back(new iwrite("s2", "im4"));
   prc.push_back(new print_("im4"));
   prc.push_back(new print_("im2"));
   prc.push_back(new add_("im2", "im3", "im5"));
   prc.push_back(new iread("s1", "flag", "im0"));
   prc.push_back(new inv("im0", "im1"));
   prc.push_back(new inv("im3", "im4"));
   prc.push_back(new inv("M", "im7"));
   prc.push_back(new print_("im0"));

   prc.push_back(new str_("im1.jpg", "s1"));
   prc.push_back(new str_("im2.jpg", "s2"));
   prc.push_back(new mat_("480 640 16", "M"));
   prc.push_back(new num_("0", "flag"));
   prc.push_back(new num_("0.7", "x"));
   prc.push_back(new num_("0.7", "y"));
   prc.push_back(new num_("0.7", "z"));
   prc.push_back(new sub_("im0", "im1", "im2"));

   auto prnt = [&]() {
      for (const auto pc: prc) {
         cerr << pc->str() << endl;
      }
   };

   sort(prc);
   sort(prc);
   cerr << "post sort2" << endl;
   prnt();

   int tainted = sanity(prc);
   if (!tainted) {
      for (const auto pc: prc) {
         pc->process(u);
      }
   }
   for (auto v: u) {
      cerr << v.first << " " << v.second.str() << endl;
   }
   return 0;
}
