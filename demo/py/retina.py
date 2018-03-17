import cv2, numpy as np
inputImage = cv2.imread('checkershadow_illusion4med.jpg', 1)
retina = cv2.bioinspired.createRetina((inputImage.shape[1], inputImage.shape[0]))
# the retina object is created with default parameters. If you want to read
# the parameters from an external XML file, uncomment the next line
#retina.setup('MyRetinaParameters.xml')
# feed the retina with several frames, in order to reach 'steady' state
for i in range(2):
    retina.run(inputImage)
# get our processed image :)
retinaOut_parvo = retina.getParvo()
# show both the original image and the processed one
cv2.imshow('image', inputImage)
cv2.imshow('retina parvo out', retinaOut_parvo)
print(np.sum(retinaOut_parvo))
# wait for a key to be pressed and exit
cv2.waitKey(0)
cv2.destroyAllWindows()
# write the output image on a file
cv2.imwrite('checkershadow_parvo.png', retinaOut_parvo)
