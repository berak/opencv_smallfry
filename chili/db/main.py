#!/usr/bin/env python

import logging, urllib

from google.appengine.ext import db
#from google.appengine.api import memcache
from google.appengine.ext import webapp
from google.appengine.ext.webapp.util import run_wsgi_app

class CodeEntry(db.Model):
    share = db.StringProperty()
    img   = db.StringProperty()
    code  = db.TextProperty()

class UpHandler(webapp.RequestHandler):
  def post(self):
    ce = CodeEntry()
    ce.share = self.request.get("key")
    ce.code  = self.request.get("code")
    ce.img   = self.request.get("img")
    logging.info( "up " + str(ce.share) + " : "+ str(ce.img) + " : " + str(ce.code) )
    #memcache.set(key, (img_url,code))
    ce.put()
    self.response.out.write('\r\n\r\nthanks\r\n')


class DnHandler(webapp.RequestHandler):
  def get(self, resource):
    share = urllib.unquote(resource)
    #img_url,code = memcache.get(urllib.unquote(resource))
    ce = db.GqlQuery("SELECT * FROM CodeEntry WHERE share='"+share+"'")
    ce = ce[0]
    self.response.out.write(ce.img + '\r\n' + ce.code + '\r\n')


def main():
  application = webapp.WSGIApplication(
    [('/up',UpHandler),
     ('/([^/]+)?', DnHandler),
    ], debug=True)
  run_wsgi_app(application)

if __name__ == '__main__':
  main()
