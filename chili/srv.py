#
# what is it ? an interactive opencv c++ (and java !) compiler.
#
#   this script heavily depends on github.com.berak.sugarcoatedchili/bin/compile
#   base assumptions:
#      local (static) openv installs for 2.4(ocv) and 3.0(ocv30) were extracted
#      ant (for java) was downloaded and extracted

import sys, socket, threading, time, datetime, os, random
import subprocess, urllib, urllib2
from cgi import parse_qs, escape
from wsgiref.simple_server import make_server
try:
    from io import BytesIO
except ImportError:
    from StringIO import StringIO as BytesIO


code_java_static="""
class SimpleSample {
    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }
    public static void cout(Object ... s){ for(Object z:s)System.out.println(z); }
    public static void cerr(Object ... s){ for(Object z:s)System.err.println(z); }
    public static void help(){ cerr("help(classname,item);\\n  'classname' should be canonical, like org.opencv.core.Mat\\n  'item' can be: CONSTRUCTOR, FIELD, METHOD, CLASS, ALL"); }
    public static void help(String cls){ ClassSpy.reflect(cls,"CLASS"); }
    public static void help(String cls,String item){ ClassSpy.reflect(cls,item); }
    public static void main(String[] args) {
"""
code_java_pre_24="""
import java.util.*;
import org.opencv.core.*;
import org.opencv.highgui.*;
import org.opencv.imgproc.*;
import org.opencv.video.*;
import org.opencv.objdetect.*;
""" + code_java_static + """
        Mat ocv = Highgui.imread("input.img",-1);
        if ( ocv.empty() )
            ocv = new Mat(8,8,CvType.CV_8UC3,new Scalar(40,40,40));
"""
code_java_pre_30="""
import java.util.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.*;
import org.opencv.video.*;
import org.opencv.objdetect.*;
import org.opencv.features2d.*;
""" + code_java_static + """
        Mat ocv = Imgcodecs.imread("input.img",-1);
        if ( ocv.empty() )
            ocv = new Mat(8,8,CvType.CV_8UC3,new Scalar(40,40,40));
"""
code_java_post_24="""
        ;;
        Highgui.imwrite("output.png", ocv);
        System.exit(0); // to break out of the ant shell.
    }
}
"""
code_java_post_30="""
        ;;
        Imgcodecs.imwrite("output.png", ocv);
        System.exit(0); // to break out of the ant shell.
    }
}
"""

code_cpp_pre_static="""
using namespace cv;
#include <algorithm>
#include <iostream>
#include <numeric>
#include <bitset>
#include <map>
using namespace std;
void download(const char * url, const char * localthing) {
    system(format("curl -s -o %s '%s'",localthing,url).c_str());
}
Mat urlimg(const char * url) {
    download(url,"local.img");
    Mat im = imread("local.img", -1);
    //system("rm local.img");
    return im;
}
int main()
{
    Mat ocv = imread("input.img",-1);
    if ( ocv.empty() )
        ocv = Mat(8,8,CV_8UC3,Scalar(40,40,40));
"""

code_cpp_pre_24="""
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/video/background_segm.hpp"
""" + code_cpp_pre_static

code_cpp_pre_30="""
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/xobjdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/face.hpp"
#include "opencv2/bgsegm.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/shape.hpp"
#include "opencv2/saliency.hpp"
#include "opencv2/text.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/xphoto.hpp"
#include "opencv2/video.hpp"
""" + code_cpp_pre_static

code_cpp_post="""
    ;;
    imwrite("output.png", ocv);
    return 0;
}
"""

style="""
<style>
    body,iframe,textarea,table,input,button,select,option,scrollbar,div{
      font-family: Arial, "MS Trebuchet", sans-serif;  font-size: 12;
      background-color: #333;   color:#aaa;
      border-color:#777;  border-style:solid;  border-width:2;
      margin: 5 5 5 5;
    }
    a{ text-decoration: none;  color:#888; }
    a:hover{  color:#ddd; }
    body{  margin: 15 15 15 15;  border: 0; }
    textarea,pre{  font-family: Lucida Console; }
</style>
"""


z_js = """
    var canvas = document.getElementById('input_url');
    canvas.onmousemove = function (evt) {
        canvas.title = '(' + (evt.clientX - canvas.x) + ',' + (evt.clientY - canvas.y) + ')'
    }    
"""


def write_faq():
    faq = [
        ["what is it ?", "an online opencv c++ / java compiler,<br> meant as an interactive pastebin,<br> or a quick tryout without installing anything locally.<br> basically, your code is running inside some shim, like int main(){/*your code*/}"],
        ["what can i do ?", "e.g. load an image into ocv, manipulate it, show the result."],
        ["any additional help ?", "<a href=answers.opencv.org>answers.opencv.org</a>, <a href=docs.opencv.org>docs.opencv.org</a>, #opencv on freenode"],
        ["opencv version ?", "2.4.9 / 3.0.0."],
        ["do i need opencv installed ?", "no, it's all in the cloud.<br>minimal knowledge of the opencv c++/java api is sure helpful."],
        ["no video ?", "no, unfortunately. you can download / manipulate exactly 1 image only (the one named 'ocv')"],
        ["is there gpu support of any kind, like ocl or cuda ?", "none of it atm. <br>(heroku even seems to support ocl, but i'm too lazy to try that now.)"],
        ["does it do c++11 ?", "it supports -std=c++0x only.<br>we're running on g++ (Ubuntu 4.4.3-4ubuntu5.1) 4.4.3."],
        ["where are the cascades ?", "in './ocv/share/OpenCV/haarcascades', './ocv/share/OpenCV/lbpcascades'"],
        ["examples ?", "Mat m = Mat::ones(4,4,5);\r\ncerr << m << endl;"],
        ["i want to program in c.","oh, no.(but try java ;)"],
        #["src code ?","https://github.com/berak/opencv_smallfry/tree/master/chili"],
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
        c = urllib2.urlopen(u)
        img = c.read()            
    except: return ''
    ext = u.split(".")
    fn="input.img"
    f=open(fn,"wb")
    f.write(img)
    f.close()
    return fn

#
# .layout
#
# url              | images
# code             | prog stdout/stderr
# form buttons     | compile results
#                  | help link
#
def write_page( code, result, link='',img='',input_url='' ):
    data = '<html><head>\n'
    data += style
    data += '</head><body><table border=0 width="100%"><tr><td>\n'
    data += '<form action="/run" method=post name="f0">\n'
    data += '<textarea rows=1 cols=80 id="url" name="url" title="you can load an image (from an url) into the predefined Mat ocv here">%s</textarea>\n' % input_url
    data += '<textarea rows=35 cols=80 id="txt" name="txt" title="Mat \'ocv\' is predefined, it will get loaded and shown.">\n'
    data += code
    data += '</textarea><br>\n'
    data += '<input type=submit value="run 2.4" id="run24">\n'
    data += '<input type=button value="run 3.0" id="run30" onClick="document.f0.action=\'/run30\';document.f0.submit();">\n'
    if link: data +='&nbsp;&nbsp;&nbsp;<a href="%s">%s</a>\n' % (link,link)
    data += '</form></td><td>\n'
    data += "<b><a href='/faq' style='color: #666; font-size: 16;' title='what is this ?'>?</a></b><br><br>"
    if input_url: data += "<img src='" + input_url + "' title='"+input_url+"' id='input_url'>&nbsp;"
    data += img
    data += result
    data += '</td></tr></table>\n</body>\n'
    data += '<script>'+z_js+'</script></html>\n'
    return data


def _remove(x): 
    try: 
        os.remove(x)
    except: pass


#
# execute bot_cmd, return piped stdout/stderr
# 
def run_prog( bot_command ):
    try:
        bot = subprocess.Popen(
            bot_command,
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

    data  = collect(bot.stdout)
    data += collect(bot.stderr)
    data  = data.replace("<","&lt;").replace(">","&gt;")
    bot.wait()
    return '<pre>' + data + '</pre>'



def run_cpp( code, v30 ):
    # save code
    f = open("cv.cpp","wb")
    if v30:
        f.write(code_cpp_pre_30)
    else:
        f.write(code_cpp_pre_24)
    f.write(code)
    f.write(code_cpp_post)
    f.close()

    # start bot
    script = "bash build.cv.sh"
    if v30: script = "bash build.cv.30.sh"
    data  = run_prog( script )
    data += "<hr NOSHADE>"
    data += run_prog( "./cv" )
    _remove("cv")
    return data


def run_java( code,v30 ):
    # save code
    f = open("src/SimpleSample.java","wb")
    if v30:
        f.write(code_java_pre_30)
        f.write(code)
        f.write(code_java_post_30)
    else:
        f.write(code_java_pre_24)
        f.write(code)
        f.write(code_java_post_24)
    f.close()
    _remove("output.png")

    # start (ant) bot
    script = "bash build.java.sh"
    if v30: script = "bash build.java.30.sh"
    return run_prog( script )


def check_code(code):
    if code.find("System.") >=0  : return "java"
    if code.find("Core.")   >=0  : return "java"
    if code.find("Highgui.")>=0  : return "java"
    if code.find("Imgproc.")>=0  : return "java"
    if code.find("org.opencv.")>=0  : return "java"
    return "cpp"


#
# top level page choices:
# 
#/
#/faq
#/run
#/share
#/output.png
#
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
    elif url.startswith(b'/share') :
        try:
            key = url.split("/")[2]
            txt = urllib2.urlopen("http://sugarcoatedchili.appspot.com/"+key).read()
            lll = txt.find('\r\n')
            input_url = txt[0:lll]
            code = txt[lll+2:]
        except: pass
        data = write_page(code,'',url,'',input_url)
    elif url.startswith(b'/run') :
        _remove("output.png")
        key = str(int(random.random()*100000))
        dat = urllib.urlencode({"key":key,"code":code,"img":input_url})
        req = urllib2.Request("http://sugarcoatedchili.appspot.com/up",dat)
        res = urllib2.urlopen(req)
        #res.read() # !! reading the thank_you msg will cost ~5secs extra.
        if input_url:
            _remove(input_img)
            input_img = url_image(input_url)
        lang = check_code(code)
        v30  = url.find("30") > 0
        if lang == "cpp":
            result = run_cpp(code,v30)
        if lang == "java":
            result = run_java(code,v30)
        data = write_page(code, result, "/share/" + key, '<img src="output.png" title="Mat ocv(here\'s your output)">', input_url)
        _remove("input.img")
    elif url == '/output.png' or url == '/share/output.png' :
        try:
            f = open('output.png','rb')
            data = f.read()
            f.close()
            content = "image/png"
        except: pass
    start_response( err, [ ("Content-Type", content), ("Content-Length", str(len(data))) ] )
    return iter([data])

    
httpd = make_server( '0.0.0.0', int(os.environ.get("PORT", 9000)), application )
while True: httpd.handle_request()
