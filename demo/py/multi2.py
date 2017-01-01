import multiprocessing,os

class Consumer(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        #other initialization stuff

    def run(self):
        count = 0
        while True:
            inp = self.task_queue.get()
            #Do computations on image
            count += 1
            print "consumer:", count, str(inp)
            self.result_queue.put(str(count))

        return


if __name__ == '__main__':
    tasks = multiprocessing.Queue()
    results = multiprocessing.Queue()
    consumer = Consumer(tasks,results)
    consumer.start()

    C=13
    print "hello", C
    while 1:
        C += 1
        if tasks.empty():
            tasks.put(C)
        else:
            text = tasks.get()
            print "out", text
        #os.sleep(1)

    consumer.stop()
