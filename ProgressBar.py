from time import sleep
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
        self.run()
        
    def run(self):
        def run():
            lastValue = -1
            while True:
                message = self.message
                if(message == "stop"):
                    g.ui.setProgress()
                    return False
                value = 0
                for counter in self.counters:
                    value += counter.value
                # Only update UI if the change in percent is worth it
                if(value == self.total or round((value/max(1, self.total))*300) != round((lastValue/max(1, self.total))*300)):
                    g.ui.setProgress(value, self.total, message)
                lastValue = value
                sleep(1/60)
        thread = Thread(target=run, args=())
        thread.start()
        
    def counter(self, i=0):
        return self.counters[i]
        
    def setMessage(self, message, increment=False):
        self.message = message
        if(increment):
            self.counter().value += 1
        
    def stop(self):
        self.setMessage("stop")
