#
# what is it ? an interactive opencv c++ compiler.
#

import sys, socket, threading, time, datetime, os
import subprocess, urllib2
from cgi import parse_qs, escape
from wsgiref.simple_server import make_server
try:
    from io import BytesIO
except ImportError:
    from StringIO import StringIO as BytesIO


shares = {} # /share/172635 : [image_url, string with c++ code]

code_pre="""
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core.hpp"
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
using namespace cv;

#include <algorithm>
#include <iostream>
#include <bitset>
#include <set>
#include <map>
using namespace std;

int main()
{
    Mat ocv(8,8,CV_8UC3,Scalar(40,40,40));
"""

code_comm = """
// ocv = Mat(100,100,CV_8UC3,Scalar(0,200,200));
// cvtColor(ocv,ocv,CV_BGR2HSV);
// cout << ocv.type() << " " << ocv.rows << endl;
"""

code_post="""
    imwrite("output.png", ocv);
    return 0;
}
"""

style="""
<style>
    body,iframe,textarea,table,input,button,select,option,scrollbar,input[type="file"],div,.but{
      font-family: Arial, "MS Trebuchet", sans-serif;  font-size: 12;
      background-color: #292929;   color:#aaa;
      border-color:#777;  border-style:solid;  border-width:2;
      margin: 5 5 5 5;
      -moz-appearance: none;
    }
    a{ text-decoration: none;  color:#888; }
    a:hover{  color:#ddd; }
    select { text-indent: 0.01px;    text-overflow: ''; editable: true; 
    body{  margin: 15 15 15 15;  border: 0; }
    textarea,pre{  font-family: Lucida Console; }
</style>
"""


def write_faq():
    faq = [
        ["what is it ?", "an online opencv c++ compiler,<br> meant as an interactive pastebin,<br> or a quick tryout without installing anything locally."],
        ["what can i do ?", "e.g. load an image into ocv, manipulate it, show the result."],
        ["does it stay alive ?", "for some short time. (heroku will shut it down after some minutes. there is no database, so be quick !)"],
        ["any additional help ?", "<a href=answers.opencv.org>answers.opencv.org</a>, <a href=docs.opencv.org>docs.opencv.org</a>, #opencv on freenode"],
        ["opencv version ?", "2.4.9."],
        ["do i need opencv installed ?", "no, it's all in the cloud.<br>minimal knowledge of the opencv c++ api is sure helpful."],
        ["no video ?", "no, unfortunately. you can download / manipulate exactly 1 image only (the one named 'ocv')"],
        ["is there gpu support of any kind, like ocl or cuda ?", "none of it atm. <br>(heroku even seems to support ocl, but i'm too lazy to try that atm.)"],
        ["does it do c++11 ?", "it supports -std=c++0x only.<br>we're running on g++ (Ubuntu 4.4.3-4ubuntu5.1) 4.4.3."],
        ["where are the cascades ?", "in './ocv/share/OpenCV/haarcascades', './ocv/share/OpenCV/lbpcascades'"],
        ["i want to program in c.","oh, no.<br>"],
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
    data += '<form action="/run" method=post>\n'
    data += '<textarea rows=1 cols=80 id="url" name="url" title="you can load an image (from an url) into the predefined Mat ocv here">%s</textarea>\n' % input_url
    data += '<textarea rows=35 cols=80 id="txt" name="txt" title="Mat \'ocv\' is predefined, it will get loaded and saved.">\n'
    data += code
    data += '</textarea><br>\n<input type=submit value="run" id="run">\n'
    if link: data +='&nbsp;&nbsp;&nbsp;<a href="%s">%s</a>\n' % (link,link)
    data += '</form></td><td>\n'
    data += "<b><a href='/faq' style='color: #666; font-size: 16;' title='what is this ?'>?</a></b><br><br>"
    if input_url: data += "<img src='" + input_url + "' title='"+input_url+"'>&nbsp;<!--heiliger st florian, verschon mein haus, zuend andre an-->"
    data += img
    data += result
    data += '</td></tr></table>\n</body></html>\n'
    return data



#
# execute bot_cmd, return piped stdout/stderr
# 
def run_prog( bot_command ):
    try:
        bot = subprocess.Popen(
            bot_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
    except: pass

    def collect(out):
        d=''
        while 1:
            z = out.readline()
            if not z : return d
            d +=  z 

    data = ''
    data += collect(bot.stdout)
    data += collect(bot.stderr)
    data = data.replace("<","&lt;").replace(">","&gt;")
    bot.wait()
    return '<pre>' + data + '</pre>'


def _remove(x): 
    try: 
        os.remove(x)
    except: pass


def run_code( code, input_img='' ):
    # save code
    f = open("cv.cpp","wb")
    f.write(code_pre)
    if input_img:
        f.write("   ocv = imread(\"%s\",-1);" % input_img )
    f.write(code)
    f.write(code_post)
    f.close()

    # start bot
    build_log = run_prog( "bash build.cv.sh" )
    data  = run_prog( "./cv" )
    data += "<hr NOSHADE>"
    data += build_log
    _remove("cv")
    return data



# global share_id
start_id=0

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
    global start_id
    url = environ['PATH_INFO'];

    # the environment variable CONTENT_LENGTH may be empty or missing
    try:
       request_body_size = int(environ.get('CONTENT_LENGTH', 0))
    except (ValueError):
       request_body_size = 0
    request_body = environ['wsgi.input'].read(request_body_size)
    d = parse_qs(request_body)
    print d
    code = ''
    input_img = ''
    input_url = ''
    share_url = ''
    if len(d): 
        code = d.get('txt', '')
        if code: code = code[0]
        input_url = d.get('url', '')
        if input_url: input_url = input_url[0];

    err = "404 SORRY"
    data = "<br><p><br><h5>Sorry, we could not retrieve %s.</h5>" % url
    content = "text/html"
    if url == "/":
        data = write_page('','')
        err = "200 OK"
    elif url == "/faq":
        data = write_faq()
        err = "200 OK"
    elif url.startswith(b'/share') :
        try:
            input_url, code = shares[url]
        except: pass
        data = write_page(code,'',url,'',input_url)
        err = "200 OK"
    elif url.startswith(b'/run') :
        start_id += 1
        share_url = "/share/%04d" % start_id
        shares[share_url] = [input_url, code]
        if input_url:
            _remove(input_img)
            input_img = url_image(input_url)
        result = run_code(code, input_img)
        data = write_page(code, result, share_url, '<img src="output.png" title="Mat ocv">', input_url)
        err = "200 OK"
    elif url == '/output.png' or url == '/share/output.png' :
        f=open('output.png','rb')
        data = f.read()
        f.close()
        content = "image/png"
        err = "200 OK"
    start_response( err, [ ("Content-Type", content), ("Content-Length", str(len(data))) ] )
    return iter([data])

    
httpd = make_server( '0.0.0.0', int(os.environ.get("PORT", 9000)), application )
while True: httpd.handle_request()
