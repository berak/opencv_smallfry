import cv2
import numpy as np
import pprint

mylist = []
loaded = cv2.imreadmulti(mats = mylist, filename = "2page.tiff", flags = cv2.IMREAD_ANYCOLOR )
print(loaded)
print(len(mylist))
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(mylist)



