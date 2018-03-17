import cv2
import numpy as np
import random
import pdb

# https://github.com/opencv/opencv/pull/8004

data = list()
classes = list()
test = np.array([[1,2,3,4,5]]).astype(np.float32)

for i in range(20):
    entry = list()
    for j in range(5):
        entry.append(int(random.random() * 10))
    data.append(entry)
data = np.array(data).astype(np.float32)

for i in range(20):
    classes.append(int(random.random() * 100) % 3 )
classes = np.array(classes)

#model = cv2.ml.SVM_create()
model = cv2.ml.RTrees_create()
model.train(data , cv2.ml.ROW_SAMPLE , classes)
eval_1 = model.predict(test)[1]
print(eval_1)

#model.save("SVM_test.xml")
model.save("RTrees_test.xml")
model2 = cv2.ml.RTrees_load("RTrees_test.xml")
#model2 = cv2.ml.SVM_load("SVM_test.xml")
eval_2 = model2.predict(test)[1]

print (eval_2 == eval_1)
