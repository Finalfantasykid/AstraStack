from time import sleep
import math
from threading import Thread
from multiprocessing import Manager
from Globals import g

class ProgressBar:

    manager = None    
    
    def __init__(self):
        if(ProgressBar.manager is None):
            ProgressBar.manager = Manager()
        self.counters = []
        # Have 1 counter per process to avoid using locks, which kills performance
        for i in range(0, g.nThreads):
            self.counters.append(ProgressBar.manager.Value('i', 0))
        self.total = 0
        self.message = ""
        self.start()
        
    def start(self):
        def run():
            lastValue = -1
            while True:
                message = self.message
                # Stop if message is now stop
                if(message == "stop"):
                    g.ui.setProgress()
                    return False
                sleep(1/60)
                value = 0
                if(message != ""):
                    for counter in self.counters:
                        value += counter.value
                    # Only update UI if the change in percent is worth it
                    if((value == 0 and lastValue != 0) or (value == self.total and lastValue != self.total) or value != lastValue):
                        g.ui.setProgress(value, self.total, message)
                    lastValue = value
        thread = Thread(target=run, args=())
        thread.start()
        
    def counter(self, i=0):
        return self.counters[i]
        
    def setMessage(self, message, increment=False):
        self.message = message
        if(increment):
            self.counter().value += 1
        
    def stop(self):
        sleep(1/30)
        g.ui.setProgress()
        self.setMessage("stop")
        
class ProgressCounter:
    
    def __init__(self, counter, nThreads):
        self.counted = 0
        self.counter = counter
        self.nThreads = nThreads
        
    def count(self, i, size):
        # only increment when the value will actually make a difference visually in the progress bar
        # the progress bar is probably around 300-320px, so 400 should be a safe number
        size = math.ceil(size/(400/self.nThreads)) 
        if(i % size == 0 and i != 0):
            self.counter.value += size
            self.counted = 0
        self.counted += 1
        
    def countExtra(self):
        self.counter.value += self.counted
        
