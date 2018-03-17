import numpy as np
from cv2 import *

tiger = imread('tiger.jpg')
nature = imread('nature.jpg')

mul_img = 0.5*tiger + 0.5*nature
div_img = tiger/2 + nature/2

print(mul_img.dtype)
print(div_img.dtype)
d = sum(mul_img - div_img)
print(d)

imshow('mul_image', mul_img)
imshow('div_image', div_img)
imshow('tigger', np.asarray(mul_img,np.uint8))
imshow('nature', nature)
waitKey(0)
destroyAllWindows()
