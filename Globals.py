import copy

class Object():
    def actualColor(self):
        return (self.colorMode or self.guessedColorMode)

g = Object()

g.TESTING = False
g.ui = None
g.pool = None

def cloneGlobals():
    ui = g.ui
    pool = g.pool
    g.ui = None
    g.pool = None
    gCopy = copy.deepcopy(g)
    g.ui = ui
    g.pool = pool
    return gCopy
    

