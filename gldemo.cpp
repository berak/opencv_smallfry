#include <GL/gl.h>
#include <GL/glu.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

using namespace std;
using namespace cv;

void perspective(float fov, float np, float fp) 
{  
    float vp[4];
    glGetFloatv( GL_VIEWPORT, vp );
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective( fov, float(vp[2])/vp[3], np, fp );
    //gluPerspective( fov, 1, np, fp );
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


typedef Vec3d vertex;
static vertex cube[6*4] = //6*4 
{
    { +1, -1, -1 }, { -1, -1, -1 }, { -1, +1, -1 }, { +1, +1, -1 },
    { +1, +1, -1 }, { -1, +1, -1 }, { -1, +1, +1 }, { +1, +1, +1 },
    { +1, -1, +1 }, { +1, -1, -1 }, { +1, +1, -1 }, { +1, +1, +1 },
    { -1, -1, -1 }, { -1, -1, +1 }, { -1, +1, +1 }, { -1, +1, -1 },
    { +1, -1, +1 }, { -1, -1, +1 }, { -1, -1, -1 }, { +1, -1, -1 },
    { -1, -1, +1 }, { +1, -1, +1 }, { +1, +1, +1 }, { -1, +1, +1 } 
};

typedef std::vector<vertex> vertices;
struct R {
    vertices coords;
    double scale, ps;
    int mode;
    int t;
    Vec3d pos;

    R() : scale(0.1), ps(8), mode(GL_POINTS), t(0), pos(0,0,-0.5) 
    {
    }

    void render()
    {
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        perspective(70,.1,100);

        glLoadIdentity();
        glTranslated(pos[0],pos[1],pos[2]);
        glRotatef( t/4+55, 1, 0, 0 );
        glRotatef( 45, 0, 1, 0 );
        glRotatef( 0, 0, 0, 1 );
        
        glPointSize(ps);
        glBegin(mode);
        int stride = mode==GL_TRIANGLES ? 3
                   : mode==GL_QUADS ? 4
                   : 1;
        int N = (int)coords.size();
        for (int i = 0; i < N; i+=stride) {
            glColor3ub( 80+i*0.02, 30+i*0.03, 40+i*0.02 );
            glVertex3d(scale * coords[i][0], scale * coords[i][1], scale * coords[i][2]);
            if (mode==GL_POINTS) continue;
            glVertex3d(scale * coords[i+1][0], scale * coords[i+1][1], scale * coords[i+1][2]);
            glVertex3d(scale * coords[i+2][0], scale * coords[i+2][1], scale * coords[i+2][2]);
            if (mode==GL_TRIANGLES) continue;
            glVertex3d(scale * coords[i+3][0], scale * coords[i+3][1], scale * coords[i+3][2]);

        }
        glEnd();
    }
    void initCube()
    { 
        coords.clear();
        for (int i=0; i<6*4; i++)
        {
            coords.push_back(cube[i]);
        }
        cerr << coords.size() << " cube verts." << endl;
    }
    void initObj(String fn="chess.obj", int skip=4)
    {
        ifstream obj(fn);
        char s[500];
        for (int i=0; i<skip; i++)
        {
            obj.getline(s,500);
            cerr << s << endl;
        }
        coords.clear();
        vector<Vec3i> faces;
        while(obj.good())
        {
            char c;
            obj >> c;
            if (c=='v')
            {
                double x,y,z;
                obj >> x >> y >> z;
                vertex v = {x,y,z};
                coords.push_back(v);
            }
            if (c=='f')
            {
                int x,y,z;
                obj >> x >> y >> z;
                Vec3i f = {x-1,y-1,z-1};
                faces.push_back(f);
            }
        }
        obj.close();
        cerr << coords.size() << " " << fn << " verts pre." << endl;
        vector<vertex> newc;
        for (size_t i=0; i<faces.size(); i++)
        {
            Vec3i f = faces[i];
            newc.push_back(coords[f[0]]);
            newc.push_back(coords[f[1]]);
            newc.push_back(coords[f[2]]);
        }
        coords = newc;
        cerr << coords.size() << " " << fn << " verts post." << endl;
    }
};

void on_opengl(void* param)
{
    R &r = *((R*)param);
    r.render();
}

int main(int argc, char **argv)
{
    String obj="";
    if (argc>1) obj = argv[1];
    
    R r;
    if (! obj.empty())
    {
        r.initObj(obj);
    }
    else 
    {
        r.initCube();    
        r.scale*= 0.1;
    }

    String name = "ogl";
    namedWindow(name, WINDOW_OPENGL);  
    setOpenGlDrawCallback(name, on_opengl, &r);
    while (1)
    {
        r.t++;
        updateWindow(name);
        int k = waitKey(20);
        if (k==27) break;
        if (k=='q') r.pos[2] -= 0.1;
        if (k=='w') r.pos[2] += 0.1;
        if (k=='a') r.pos[1] -= 0.1;
        if (k=='s') r.pos[1] += 0.1;
        if (k=='y') r.pos[0] -= 0.1;
        if (k=='x') r.pos[0] += 0.1;
        if (k=='3') r.mode=GL_TRIANGLES;
        if (k=='2') r.mode=GL_QUADS;
        if (k=='1') r.mode=GL_POINTS;
        if (k=='-') r.scale *= 0.99;
        if (k=='+') r.scale *= 1.1;
        if (k=='P') r.ps += 0.1;
        if (k=='p') r.ps -= 0.1;
        if (k=='c') r.initObj(obj);
        if (k=='C') r.initCube();
    }   
    return 0;
}
/*

UMat filter(const UMat &in, int i)
{
    return in.row(i).reshape(1,3);
}
UMat image(int label)
{
    UMat in(6,6,CV_32F);
    randn(in, (label > 0 ? 0.75 : 0.25), 0.25);
    return in;
}

int main()
{
    cerr << getBuildInformation() << endl;
    return 1;
    
    UMat fil(5,3*3,CV_32F);
    randu(fil,0,1);
    for (int g=0; g<10; g++)
    {
        int lab = theRNG().uniform(0,2);
        UMat in = image(lab);
        vector<UMat> preds;
        for (int r=0; r<fil.rows; r++)
        {
            UMat p;
            UMat f = filter(fil, r);
            preds.push_back(p);
            cerr << g << " >> " << r << (lab>0?" F ":" f ") << f.size() << " " << sum(f)[0] << " p " << p.size() << " " << sum(p)[0] << endl; //p.getMat(ACCESS_READ) << endl;
        }

        UMat hashed(in.size(), in.type(), 0.0f);
        for (int r=0; r<fil.rows; r++)
        {
            UMat p;
            UMat f = filter(fil, r);
            filter2D(preds[r], p, -1, f.t());
            threshold(p, p, 0.5, 1.0/(1<<(8-r)), 0);
            add(p,hashed,hashed);
            cerr << g << " << " << r << (lab>0?" F ":" f ") << f.size() << " " << sum(f)[0] << " p " << p.size() << " " << sum(p)[0] << endl; //p.getMat(ACCESS_READ) << endl;
        }
        //cerr << hashed << endl;
        UMat d;
        subtract(in,hashed,d);
        cerr << sum(d)[0] << endl;        
    }
    return 0;   
}
*/