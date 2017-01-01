import cv2
import numpy as np
import multiprocessing

class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        #other initialization stuff

    def run(self):
        count = 0
        while True:
            image = self.task_queue.get()
            #Do computations on image
            count += 1
            print "consumer:", count, np.shape(frame)
            self.result_queue.put(str(count))

        return



if __name__ == '__main__':
    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    consumer = Consumer(tasks,results)
    consumer.start()
    help(cv2.putText)
    #Creating window and starting video capturer from camera
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    while vc.isOpened():
        rval, frame = vc.read()
        if rval:
            if tasks.empty():
                tasks.put(frame)
            else:
                text = tasks.get()
                #Add text to frame
                # cv2.putText(frame,text,(60,60),cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
                print text
            #Showing the frame with all the applied modifications
            #cv2.imshow("preview", frame)
        k = cv2.waitKey(13) & 0xff
        if k==27: break

    consumer.stop()
