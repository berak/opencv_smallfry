#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <opencv2/opencv.hpp>
#include "xui.h"

using namespace std;
using namespace cv;


struct XUI : UI
{
    struct Item
    {
        Rect r;
        Scalar c;
        String name;
        float focus;
        int *v;
        
        virtual void draw(Mat &img) = 0;
        virtual void mouse(Point p, int b, int s) = 0;
        
        void textbox(Mat &img, float lev, float w=0.25f)
        {
            int s = r.height/2;
            int y = r.y + s;
            //rectangle(img,r,c*0.15,-1);
            line(img, Point(r.x + s/2, y), Point(r.x - s/2 + r.width, y), c*lev, r.height*w, CV_AA);
            putText(img,name,Point(r.x+r.width*0.3,y+3),FONT_HERSHEY_SIMPLEX,0.4,Scalar(30,100,30)+c*lev,1.5, CV_AA);
        }
        bool hasmouse(bool cond)
        {
            focus = cond ? 1.2f : 0.9f;
            return cond;
        }
    };
    
    vector<Ptr<Item>> its;
    bool visible;
    String name;

    XUI(String n="ui") : visible(false), name(n) 
    {
        cv::namedWindow(n,0);
        cv::setMouseCallback(n, onmouse, this);
    }

    struct Slider : Item
    {
        Slider() {mouse(Point(0,0),0,0);}
        virtual void draw(Mat &img)
        {
            int s = r.height/2;
            int y = r.y + s;
            textbox(img, focus, 0.4f);
            circle(img, Point(r.x + (*v), y), r.height/5, c*1.5*focus, 2, CV_AA);
        }
        virtual void mouse(Point p, int b, int s)
        {
            if (! hasmouse(s == 1 && r.contains(p))) return;
            int x = p.x - r.x;
            *v = x;
        }
    };

    struct Button : Item
    {
        Button() {mouse(Point(0,0),0,0);}
        virtual void draw(Mat &img)
        {
            int s = r.height/2;
            int y = r.y + s;
            float lev = focus + *v * 0.7;
            textbox(img,lev,0.55f);
        }
        virtual void mouse(Point p, int b, int s)
        {
            if (! hasmouse(b == 1 && r.contains(p))) return;
            *v = ! *v;
        }
    };

    struct Trigger : Button
    {
        virtual void mouse(Point p, int b, int s)
        {
            *v=0;
            if (! hasmouse(b == 1 && r.contains(p))) return;
            *v=1;
        }
    };

    struct Choice : Button
    {
        vector<String>choice;

        virtual void mouse(Point p, int b, int s)
        {
            if (! hasmouse(b == 1 && r.contains(p))) return;
            int &sel = *v;
            Rect left(r); left.width/=2;
            bool inleft = left.contains(p);
            if (inleft && (sel >0))
                sel--;
            else if ((! inleft) && (sel < int(choice.size())-1))
                sel++;
            name = choice[sel];
        } 
    };

    template <typename IT>
    Ptr<IT> add(String &name, const Rect &r, Scalar c, int &v)
    {
        Ptr<IT> it = makePtr<IT>();
        it->r=r;
        it->c=c;
        it->v = &v;
        it->name = name;
        its.push_back(it);
        return it;
    }

    void addSlider(String name, const Rect &r, Scalar c, int &v)
    {    add<Slider>(name,r,c,v);    }
    void addButton(String name, const Rect &r, Scalar c, int &v)
    {    add<Button>(name,r,c,v);    }
    void addTrigger(String name, const Rect &r, Scalar c, int &v)
    {    add<Trigger>(name,r,c,v);    }
    void addChoice(vector<String> &choice, const Rect &r, Scalar c, int &v)
    {
        String n("choice");
        Ptr<Choice> ch = add<Choice>(n,r,c,v);
        ch->choice=choice;
        ch->name=choice[v];
    }

    void onmouse(int b, int x, int y, int s)
    {
        if (! visible) return;
      
        Point pt(x,y);
        for (size_t i=0; i<its.size(); i++)
        {
            its[i]->mouse(pt, b, s);
        }
    }
    static void onmouse(int b, int x, int y, int s, void *p)
    {
        XUI &ui = *((XUI*)p);
        ui.onmouse(b,x,y,s);
    }

    virtual bool toggle()
    {
        visible = ! visible;
        return visible;
    }

    virtual void draw(Mat &img) 
    {
        if (!visible) return;
        for (size_t i=0; i<its.size(); i++)
        {
            its[i]->draw(img);
        }
    }

    virtual void show(Mat &img) 
    {
        draw(img);
        imshow(name, img);
    }

    virtual String info()
    {
        String s="";
        for (size_t i=0; i<its.size(); i++)
        {
            s += format("%s:%d ", its[i]->name.c_str(), (its[i]->v?*(its[i]->v):0));
        }
        return s;
    }
    virtual void setText(int i, cv::String t) { its[i]->name = t; }
};

Ptr<UI> createUI(String n) {
    return makePtr<XUI>(n);
}
