#
# what is it ? an interactive opencv c++ (and java !) compiler.
#
#   this script heavily depends on github.com.berak.sugarcoatedchili/bin/compile
#   base assumptions:
#      local (static) openv installs for 3.0(ocv3) were extracted
#      ant (for java) was downloaded and extracted
#

import sys, socket, threading, time, datetime, os, random
import subprocess, urllib, urllib2, base64
from cgi import parse_qs, escape
from wsgiref.simple_server import make_server
try:
    from io import BytesIO
except ImportError:
    from StringIO import StringIO as BytesIO

#
# most compiled languages have a pattern like :
#   prelude your-code postlude
#
code_pre={}
code_post={}

java_inc = """
import java.lang.Math.*;
import java.util.*;
import java.awt.*;
import java.awt.image.*;
import java.io.File;
import java.io.IOException;
import javax.imageio.*;
import org.opencv.core.*;
import org.opencv.calib3d.*;
import org.opencv.dnn.*;
import org.opencv.features2d.*;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.*;
import org.opencv.ml.*;
import org.opencv.objdetect.*;
import org.opencv.photo.*;
import org.opencv.utils.*;
import org.opencv.video.*;
import org.opencv.videoio.*;
import org.opencv.face.*;
import org.opencv.aruco.*;
import org.opencv.bioinspired.*;
import org.opencv.img_hash.*;
import org.opencv.tracking.*;
import org.opencv.xfeatures2d.*;
import org.opencv.ximgproc.*;
import org.opencv.xphoto.*;
"""

code_pre["java"] = java_inc + """

class SimpleSample {
    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }
    public static void cout(Object ... s){ for(Object z:s)System.out.println(z); }
    public static void cerr(Object ... s){ for(Object z:s)System.err.println(z); }
    public static void help(){ cerr("help(classname,item);\\n  'classname' should be canonical, like org.opencv.core.Mat\\n  'item' can be: CONSTRUCTOR, FIELD, METHOD, CLASS, ALL"); }
    public static void help(String cls){ ClassSpy.reflect(cls,"CLASS"); }
    public static void help(String cls,String item){ ClassSpy.reflect(cls,item); }
    public static void main(String[] args) {
        Mat ocv = Imgcodecs.imread("input.img",-1);
        if ( ocv.empty() )
            ocv = new Mat(8,8,CvType.CV_8UC3,new Scalar(40,40,40));
"""
code_post["java"] = """
        ;;
        Imgcodecs.imwrite("output.jpg", ocv);
        System.exit(0); // break out of the ant shell.
    }
}
"""

code_pre["kotlin"] = java_inc + """

fun main(args: Array<String>) {
    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }
    var ocv = Imgcodecs.imread("input.img",-1);
        if ( ocv.empty() )
            ocv = new Mat(8,8,CvType.CV_8UC3,new Scalar(40,40,40));
"""
code_post["kotlin"] = """
        ;;
        Imgcodecs.imwrite("output.jpg", ocv);
        System.exit(0); // break out of the ant shell.
    }
}
"""

code_pre["cpp"] = """
#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/flann.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/shape.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/superres.hpp"
#include "opencv2/video.hpp"

#include "opencv2/aruco.hpp"
#include "opencv2/bgsegm.hpp"
#include "opencv2/bioinspired.hpp"
#include "opencv2/ccalib.hpp"
#include "opencv2/dpm.hpp"
#include "opencv2/face.hpp"
#include "opencv2/line_descriptor.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/rgbd.hpp"
#include "opencv2/saliency.hpp"
#include "opencv2/stereo.hpp"
#include "opencv2/structured_light.hpp"
#include "opencv2/surface_matching.hpp"
#include "opencv2/text.hpp"
#include "opencv2/tracking.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/xobjdetect.hpp"
#include "opencv2/xphoto.hpp"

using namespace cv;
#include <algorithm>
#include <iostream>
#include <chrono> // c++11 !!
#include <numeric>
#include <vector>
#include <bitset>
#include <map>
#include <set>
using namespace std;

void download(const char * url, const char * localthing) {
    int n = system(format("curl -s -o %s '%s'",localthing,url).c_str());
}
Mat urlimg(const char * url) {
    download(url,"local.img");
    Mat im = imread("local.img", -1);
    return im;
}
int main()
{
    Mat ocv = imread("input.img",-1);
    if ( ocv.empty() )
        ocv = Mat(8,8,CV_8UC3,Scalar(40,40,40));
"""
code_post["cpp"] = """\r\n;;\r\n    imwrite("output.jpg", ocv);\r\n    return 0;\r\n}\r\n"""
code_pre["py"] = "import tensorflow as tf, cv2, numpy as np\r\nocv = cv2.imread('input.img',-1)\r\n"
code_post["py"] = "\r\nif np.shape(ocv)!=():\r\n  cv2.imwrite('output.jpg',ocv)\r\n"

style="""
<link rel="icon" href="favicon.ico" type="image/x-icon" />
<style>
    body,iframe,textarea,table,input,button,select,option,scrollbar,div{
      font-family: Arial, "MS Trebuchet", sans-serif;  font-size: 13px;
      background-color: #333;   color:#aaa;
      border-color:#777;  border-style:solid;  border-width: 1px;
      margin: 5px 5px 5px 5px;
    }
    a{ text-decoration: none;  color:#888; }
    a:hover{ color:#ddd; }
    body{  margin: 15px 15px 15px 15px ;  border: 0; }
    textarea,pre{ font-family: Lucida Console; }
</style>
"""


#z_js = """
#    var canvas = document.getElementById('input_url');
#    canvas.onmousemove = function (evt) {
#        canvas.title = '(' + (evt.clientX - canvas.x) + ',' + (evt.clientY - canvas.y) + ')'
#    }
#"""


def write_faq():
    faq = [
        # ["**NEW**", "kotlin-1.1.0 support "],
        ["**NEW**", "tensorflow 0.8 support (from python)"],
        ["what is it ?", "an online opencv c++ / java / python compiler,<br> meant as an interactive pastebin,<br> or a quick tryout without installing anything locally.<br> basically, your code is running inside some shim, like int main(){/*your code*/}"],
        ["what can i do ?", "e.g. load an image into ocv, manipulate it, show the result. change something, press the run button"],
        ["any additional help on opencv ?", "<a href=answers.opencv.org>answers.opencv.org</a>, <a href=docs.opencv.org>docs.opencv.org</a>, #opencv on freenode"],
        ["do i need opencv installed ?", "no, it's all in the cloud.<br>minimal knowledge of the opencv c++/java api is sure helpful."],
        ["opencv version ?", "3.3.0 (cv2 is from PIP)"],
        ["no video ?", "no, unfortunately. you can download / manipulate exactly 1 image only (the one named 'ocv')"],
        ["does it have cuda support ?", "no. ;("],
        ["does it do c++11 ?", "it supports -std=c++0x only.<br>we're running on g++ (Ubuntu 4.4.3-4ubuntu5.1) 4.4.3."],
        ["where are the cascades ?", "in './ocv3/share/opencv4/haarcascades', './ocv3/share/opencv4/lbpcascades'"],
        ["examples ?", "Mat m = Mat::ones(4,4,5);\r\ncerr << m << endl;"],
        ["this seems to generate additional boilerplate code, where can i see, what's actually run ?", "* c++ : src/cv.cpp<br>* java : /src/SimpleSample.java <br>* python: http://sugarcoatedchili.herokuapp.com/src/ocv.py"],
    ]
    data = '<html><head>\n'
    data +='<title>faq</title>'
    data += style
    data += '</head><body><div id="titles"><ul>\n'
    for f in faq:
        k = f[0]
        data +="<li><a href='#%s'>%s</a></li>\n" % (k,k)
    data +="<p></ul></div><div id='targets'><ul>\n"
    for f in faq:
        k = f[0]
        v = f[1]
        data +="<li> <a name='%s'>%s</a></li>\n" % (k,k)
        data += v + "<br><p>\n"
    data +="</ul></div>\n"
    data += '</body></html>\n'
    return data

#
# download an image url, save it, and return the local filename
#
def url_image(u):
    try:
        c = urllib2.urlopen(u) # TODO NOT portable to py3 ! (main reason i cannot upgrade)
        img = c.read()
    except: return ''
    fn="input.img"
    f=open(fn,"wb")
    f.write(img)
    f.close()
    #hack for missing cpp/java png support
    if u.find(".png")>0:
        import cv2
        im = cv2.imread("input.img", -1)
        cv2.imwrite("input.jpg", im)
        os.rename("input.jpg", "input.img")
    return fn

def get_file(fn):
    try:
        f = open(fn,"rb")
    except: return ""
    it = f.read()
    f.close()
    return it

#
# .layout
#
# img url box      | help link
# code box         | images
# form buttons     | compile output
#                  | prog stdout/stderr

def write_page( code, result, link='',img='',input_url='' ):
    data = '<!DOCTYPE html>\n<html><head><meta charset="utf-8">\n'
    data += style
    data += '</head><body><table border=0 width="100%"><tr><td>\n'
    data += '<form action="/run" method=post name="f0">\n'
    data += '<textarea rows=1 cols=90 id="url" name="url" title="you can load an image (from an url) into the predefined Mat ocv here">%s</textarea>\n' % input_url
    data += '<textarea rows=30 cols=90 id="txt" name="txt">\n'
    data += code
    data += '</textarea><br>\n'
    data += '<input type=button value="run code" id="run" onClick="document.f0.action=\'/run\';document.f0.submit();">\n'
    if link: data +='&nbsp;&nbsp;&nbsp;<a href="%s">%s</a>\n' % (link,link)
    data += '</form></td><td>\n'
    data += "<b><a href='/faq' style='color: #666; font-size: 16;' title='what is this ?'>?</a></b><br><br>"
    if input_url: data += "<img src='" + input_url + "' title='"+input_url+"' id='input_url'>&nbsp;"
    data += img
    data += result
    data += '</td></tr></table>\n</body>\n'
    #data += '<script>'+z_js+'</script></html>\n'
    return data


def _remove(x):
    try:
        os.remove(x)
    except: pass

#
# execute shell cmd, return piped stdout/stderr
#
def run_prog( command ):
    try:
        cmd = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True)
    except: pass

    def collect(out):
        d=''
        while 1:
            z = out.readline()
            if not z : return d
            d +=  z

    data  = collect(cmd.stdout)
    data += collect(cmd.stderr)
    data  = data.replace("<","&lt;").replace(">","&gt;")
    cmd.wait()
    return '<pre>' + data + '</pre>'

def write_shim(f, code, tp):
    f.write(code_pre[tp])
    f.write(code)
    f.write(code_post[tp])

def run_cpp( code ):
    # save code
    f = open("src/cv.cpp","wb")
    if code.find("#include") > -1:
        f.write(code) # a standalone cpp file
    else: # insert code into shimmy
        write_shim(f, code, "cpp")
    f.close()
    # start g++ script
    script = "bash build.cv.3.sh"
    data  = run_prog( script )
    data += "<hr NOSHADE>"
    data += run_prog("src/cv")
    _remove("cv")
    return data

def run_java( code ):
    # save code
    f = open("src/SimpleSample.java", "wb")
    write_shim(f, code, "java")
    f.close()
    # start ant tool
    return run_prog("bash build.java.3.sh")

def run_kotlin( code ):
    # save code
    f = open("src/ocv.kt", "wb")
    write_shim(f, code, "kotlin")
    f.close()
    # start ant tool
    return run_prog("bash build.kotlin.3.sh")

def run_python( code ):
    # save code
    f = open("src/ocv.py","wb")
    write_shim(f, code, "py")
    f.close()
    # start snek
    return run_prog("python src/ocv.py")


def check_code(code):
    #if code.find("val ")       >=0 : return "kotlin"
    #if code.find("var ")       >=0 : return "kotlin"
    #if code.find("fun ")       >=0 : return "kotlin"
    if code.find("import cv2") ==0 : return "python"
    if code.find("def ")    ==0    : return "python"
    if code.find("cv2.")    >=0    : return "python"
    if code.find("np.")     >=0    : return "python"
    if code.find("tf.")     >=0    : return "python"
    if code.find("java.")   >=0    : return "java"
    if code.find("CvType.") >=0    : return "java"
    if code.find("System.") >=0    : return "java"
    if code.find("Core.")   >=0    : return "java"
    if code.find("Imgproc.")>=0    : return "java"
    if code.find("org.opencv.")>=0 : return "java"
    return "cpp"

#
# connect to redis db
#
def redis_con():
    red = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c = red.connect(("catfish.redistogo.com", 10644))
    red.send("auth 56e2f0e64f803f243047db64b8fd045e\r\n")
    m = red.recv(512) # +OK
    return red
# get code or image url
def redis_get(red, key):
    red.send("get " + str(key) + "\r\n")
    m = red.recv(8*1024)
    m = "".join(m.split("\r\n")[1:-1])
    return base64.b64decode(m)


#/
#/faq
#/run
#/share
#/output.jpg
#/js/utils.js
#/js/opencv.js
#/src/ocv.kt # latest attempt
#/src/cv.cpp # latest attempt
#/src/ocv.py # latest attempt
#/src/SimpleSample.java # latest attempt
def application(environ, start_response):
    code = ''
    input_img = ''
    input_url = ''
    share_url = ''
    url = environ['PATH_INFO'];
    try:
       request_body_size = int(environ.get('CONTENT_LENGTH', 0))
    except (ValueError):
       request_body_size = 0
    request_body = environ['wsgi.input'].read(request_body_size)
    d = parse_qs(request_body)
    if len(d):
        code = d.get('txt', '')
        if code: code = code[0]
        input_url = d.get('url', '')
        if input_url: input_url = input_url[0];
    data = ""
    err  = "200 OK"
    content = "text/html"
    if url == "/":
        data = write_page('','')
    elif url == "/faq":
        data = write_faq()
    elif url.startswith(b'/share') : # retrieve saved state from redis db
        key = url.split("/")[2]
        red = redis_con()
        code = redis_get(red,key)
        input_url = redis_get(red,str(key)+"_url")
        data = write_page(code,'',url,'',input_url)
    elif url.startswith(b'/run') : # save attempt to redis, then execute it
        _remove("output.jpg")
        key = str(int(random.random()*100000))
        red = redis_con()
        red.send("set "+key+" "+ base64.b64encode(code) +"\r\n")
        red.send("expire "+key+" 1250000\r\n") # ~ 2 weeks
        if input_url:
            red.send("set "+key+"_url "+ base64.b64encode(input_url) +"\r\n")
            red.send("expire "+key+"_url 1250000\r\n")
            _remove(input_img)
            input_img = url_image(input_url)
        lang = check_code(code)
        if lang == "cpp":
            result = run_cpp(code)
        if lang == "java":
            result = run_java(code)
        if lang == "python":
            result = run_python(code)
        #if lang == "kotlin":
        #    result = run_kotlin(code)
        data = write_page(code, result, "/share/" + key, '<img src="output.jpg" title="Mat ocv(here\'s your output)">', input_url)
        _remove("input.img")
    elif url == '/output.jpg' or url == '/share/output.jpg' :
        data = get_file('output.jpg')
        content = "image/jpg"
    elif url == '/src/SimpleSample.java' or url == '/src/cv.cpp'or url == '/src/ocv.kt' or url == '/src/ocv.py' or url == '/js/utils.js' or url=='/js/opencv.js':
        data = get_file(url[1:])
        content = "text/plain"
    else:
        data = "404 could not retrieve " + url
        err = "404 not found"
    start_response( err, [ ("Content-Type", content), ("Content-Length", str(len(data))) ] )
    return iter([data])


httpd = make_server( '0.0.0.0', int(os.environ.get("PORT", 9000)), application )
while True: httpd.handle_request()
