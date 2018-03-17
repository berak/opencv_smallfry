#include "opencv2/opencv.hpp"
#include "profile.h"
using namespace cv;
using namespace std;

inline float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-1*x));
}
inline float sigmoid_sym(float x) {
	float t = x * x * 2.751;
	return 0.5f * (1.0f - t) / (1.0f + t);
}
inline float relu(float x) {
	return (x > 0) ? x : 0; //0.01f * x;
}
inline float linear(float x) {
	return 1.0/(x*x*5+1);
}
inline float act(float x) {
	return linear(x);
}
struct Deepq {
	struct Experience {
		Mat state;
		int action;
		float reward;

		Experience(const Mat &s=Mat(), int a=0, float r=0) : state(s), action(a), reward(r)	{}
	};

	const size_t MaxExp = 1000;
	float gamma;
	int current,nactions,batchsize,age;
	vector<Experience> e;
	Ptr<ml::ANN_MLP> nn;
	RNG rng;

	Deepq(int nin, int nout, int batch) : gamma(0.05f), current(0), nactions(nout), batchsize(batch), age(0), rng(getTickCount()) {
		Mat_<int> layers(1,4); layers << nin, 256, 256, nout;
		nn = ml::ANN_MLP::create();
		nn->setLayerSizes(layers);
		nn->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.001);
		nn->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM);
		nn->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1, 0.0001));
	}

	// todo: we need independant test data, not the train set
	float loss() {
		PROFILE
		Mat data, labels(MaxExp-1, nactions, CV_32F, 0.0f);
		for (int i=1; i<MaxExp; i++) {
			data.push_back(e[i].state);
			labels.at<float>(i-1, e[i-1].action) = act(gamma*e[i-1].reward);
		}
		return nn->calcError(ml::TrainData::create(data,0,labels), false, noArray());
	}

	int forward(const Mat &state, bool learn) {
		PROFILE
		int action = 0;
		age ++;
		int n = rng.uniform(0, 10);
		if (n>3 || (!nn->isTrained())) {
		    action = rng.uniform(0, nactions);
		} else {
			action = (int)nn->predict(state);
		}
		if (1) {
			// save experience
			// (we don't know the reward yet)
			Experience ex(state, action, 0);
			if (e.size() < MaxExp) {
				e.push_back(ex);
			} else {
				e[current] = ex;
			}
		}
		return action;
	}

	void backward(float reward) {
		PROFILE
		e[current].reward = reward;
		current ++;
		current %= MaxExp;
		int flag = nn->isTrained() ? 0 : 1+2+4;
		if (e.size() < batchsize) {
			Mat labels(1, nactions, CV_32F, 0.0f);
			randu(labels, 0, 1.0/nactions);
			nn->train(ml::TrainData::create(e.back().state, 0, labels), flag);
			return;
		}
		for (int b=0; b<batchsize; b++) {
			PROFILEX("batch")
			// "dream" a random experience
			size_t i = theRNG().uniform(0, e.size()-1);
			Experience &cur = e[i];
			Experience &nxt = e[(i+1) % e.size()];
			// max policy
			float maxval=0;
			int maxid=0;
			{
				PROFILEX("policy")
				Mat_<float> res;
				nn->predict(cur.state, res);
				if ((b<3)&&(age%150==0))cerr << age << format(" {%6.3f %6.3f %6.3f}  ",res(0),res(1),res(2));
				for (int r=0; r<res.total(); r++) {
					float v = res.at<float>(r);
					if (v > maxval) {
						maxid = r;
						maxval = v;
					}
				}
			}
			// update network weights on next state
			{
				PROFILEX("train")
				Mat_<float> labels(1, nactions, 0.0f);
				float activation = act(gamma * maxval * cur.reward);
				labels(0, cur.action) = activation;
				bool ok = nn->train(ml::TrainData::create(nxt.state, 0, labels), flag); //  update
				if ((b<3) && (age%150==0))cerr << format("{%5.3f %5.3f %5.3f} %d %d %5.3f %5.3f %5.3f",labels(0),labels(1),labels(2),cur.action,maxid,maxval,cur.reward,activation) << endl;
			}
			// cost ?
		}
		if (age % 3000 == 0)
			nn->save("rlmaze.xml");
	}
};

typedef vector<Point3f> Pills; // x,y,color
Point3f newPill(const Mat &maze) {
	return Point3f{
		float(theRNG().uniform(0, maze.cols)),
		float(theRNG().uniform(0, maze.rows)),
		float(theRNG().uniform(0, 2)*2-1)
	};
}
Pills randomPills(const Mat &maze) {
	Pills pills;
	for (int i=0; i<20; i++) {
		pills.push_back(newPill(maze));
	}
	return pills;
}


struct Pacman {
	Deepq brain;
	Point pos;
	int rad, srad, angle;
	bool learn;
	Pacman() : brain(9,3,16), pos(20,120), rad(10), srad(50), angle(90), learn(true) {}

	bool sense(Mat &maze, Mat &state) {
		state = Mat(1,9,CV_32F,0.0f);
		bool can_move = true;
		//float weights[9] = {0.6, 0.7, 0.8, 1.0, 1.2, 1.0, 0.8, 0.7, 0.6}; // prefer forward
		float weights[9] = {1,1,1,1,1,1,1,1,1}; // straight
		//float weights[9] = {1,1,0.9,0.5,0.3,0.5,0.9,1,1};
		for (int k=0,i=-60; i<=60; i+=15,k++) {
		    double s = sin((i+angle)*CV_PI/180);
		    double c = cos((i+angle)*CV_PI/180);
		    Point p2(pos.x+s*srad, pos.y+c*srad);
		    LineIterator it(maze, pos, p2, 8);
		    int r=0;
		    float food = 0.0f;
		    for (; r<srad; r++) {
		        Vec3b & pixel = maze.at<Vec3b>(it.pos());
		        if (pixel[0] > 50)
		            pixel[1] = pixel[2] = 0; // set g and b to 0, leaves blue line
		        else {
		        	bool g = pixel[1] > 50;
		        	bool r = pixel[2] > 50;
		        	food = (g && !r) ? 1.0f : (!g && r) ? -1.0f : -0.3f;
		            break;
		        }
		        it++;
		    }
		    if (r<rad+3 && food<=0) can_move = false;
		    state.at<float>(0,k) = (food + weights[k]) * (float)r / srad;
		}
		return can_move;
	}

	void move(Mat &maze) {
		Mat state;
		bool can_move = sense(maze, state);
		int action = brain.forward(state, learn);
		switch(action) {
			case 0:
				if (can_move) {
				    double s = sin(angle*CV_PI/180);
				    double c = cos(angle*CV_PI/180);
				    pos = Point(pos.x+s*3, pos.y+c*3);
				}
				break;
			case 1:	angle -= 10; break;
			case 2:	angle += 10; break;
		}
		if (! learn) return;
		//float reward = can_move ? sum(state)[0] : 0.1f;
		float reward = sum(state)[0];
		brain.backward(reward);
	}

	void draw(Mat &draw) {
		circle(draw,pos,rad,Scalar(255,0,0),-1,CV_AA);
	}
};

int main() {
	Mat img = imread("maze.png");
	Pills pills = randomPills(img);
	Pacman pac;
	while(1) {
		Mat maze = img.clone();
		if ((pac.brain.age>1100) && (pac.brain.age % 500==0))
			cerr << endl << "loss: " << pac.brain.loss() << endl;
		pac.move(maze);
		pac.draw(maze);
		for (auto p=pills.begin(); p!=pills.end(); p++) {
			if (abs(pac.pos.x-p->x)<pac.rad+5+7 && abs(pac.pos.y-p->y)<pac.rad+5+7) {
				cerr << "- " << *p;
				*p = newPill(maze);
				cerr << " + " << *p << endl;
			} else {
				circle(maze,Point(p->x,p->y),7,(p->z>0?Scalar(0,0,180):Scalar(0,180,0)), -1, LINE_AA);
			}
		}
		imshow("maze", maze);
		int k = waitKey(3);
		if (k =='r') pills = randomPills(maze);
		if (k ==27) break;
		if (k=='l') { pac.learn = ! pac.learn; cerr << "learn " << pac.learn << " " << pac.brain.age << endl; }
	}
	return 0;
}
