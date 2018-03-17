#include <GL/gl.h>
#include <GL/glu.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/opengl.hpp>

using namespace cv;
using namespace std;


struct R {
    double scale, ps;
    int t;
    Vec3d pos;
    Mat_<uchar> face;
    R() : scale(0.1), t(0), pos(-5,-5,-35.8)
    {
        face = imread("../img/head2.tif",0);
    }

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

    void _renderPoint( int i, int j, float wn, float hn, int DIM)
    {
        float pz = (float)face(i,j)/8;
        float px = wn + (float)i * wn;
        float py = hn + (float)j * hn;
        //float pz = hn + (float)z * hn;
        float tx = (float)i /DIM;
        float ty = (float)j /DIM;

        //glNormal3f( 0, 1, 0 );
        glTexCoord2f( ty, tx );
        glVertex3f( px, py, pz );
    }

    void drawPlane()
    {
        int w = face.cols, h=face.rows;
        float wn = (float)w / (float)(w + 1);  /* Grid element width */
        float hn = (float)h / (float)(h + 1);  /* Grid element height */

        for (int i=0, j=0; j < w - 1; j++)
        {
            glBegin(GL_TRIANGLE_STRIP);

            _renderPoint(i, j, wn, hn, w);
            for (i = 0; i < w - 1; i++)
            {
                _renderPoint(i, j+1, wn, hn, w);
                _renderPoint(i+1, j, wn, hn, w);
            }
            _renderPoint(w-1, j+1, wn, hn, w);

            glEnd();
        }
    }
    void render()
    {
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
        glEnable(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        perspective(70,.01,100);
        glLoadIdentity();
        glTranslated(pos[0],pos[1],pos[2]);
        glScalef(.1,.1,.1);
        glRotatef(++t/10, 0, 1, 0);
        glRotatef(-90, 0, 0, 1);
        //glRotatef( 180, 1, 0, 0 );

        drawPlane();
    }
};

void on_opengl(void* param)
{
    R &r = *((R*)param);
    r.render();
}


int main(int argc, char** argv)
{
    R r;
    String name = "ogl";
    namedWindow(name, WINDOW_OPENGL);
    setOpenGlDrawCallback(name, on_opengl, &r);

    Mat face = imread("c:/data/faces/lfw40_crop/Renee_Zellweger_0010.jpg");
    ogl::Texture2D ftex(face);;
    ftex.bind();
/*
	string path = "c:/data/mdl/head.yml.gz";
    FileStorage fs(path, FileStorage::READ);
    if (! fs.isOpened())
    {
        cerr << path << " could not be loaded !" << endl;
    }
    Mat mdl,eyemask;
    fs["mdl"] >> mdl;
    fs["eyemask"] >> eyemask;
    cout << mdl.size() << mdl.type() << endl;
    cout << eyemask.size() << eyemask.type() << endl;
    Mat ch[3];
    split(mdl, ch);
    Mat_<double> depth;
    normalize(ch[1], depth, -100);
    imshow("eye", eyemask);
    imshow("head1", depth);
    Mat dimg;
    depth.convertTo(dimg,CV_8U, 255.0);
    int FS = 210;
    int off = (depth.cols-FS)/2;
    Rect roi(off,off,FS,FS);
    dimg = dimg(roi);
    cout << dimg.size() << dimg.type() << endl;

    imshow("head2", dimg);
    imwrite("head.tif", dimg);
    Mat r2; pyrDown(dimg,r2);
    imwrite("head2.tif", r2);
    Mat r3; pyrDown(r2,r3);
    imwrite("head3.tif", r3);
    waitKey();
*/
    while (1)
    {
        r.t++;
        updateWindow("ogl");
        int k = waitKey(20);
        if (k==27) break;
        if (k=='q') r.pos[2] -= 0.1;
        if (k=='w') r.pos[2] += 0.1;
        if (k=='a') r.pos[1] -= 0.1;
        if (k=='s') r.pos[1] += 0.1;
        if (k=='y') r.pos[0] -= 0.1;
        if (k=='x') r.pos[0] += 0.1;
        if (k=='-') r.scale *= 0.99;
        if (k=='+') r.scale *= 1.1;
    }

    return 0;
}
