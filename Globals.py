import copy

class Object():
    pass

g = Object()

g.TESTING = False
g.ui = None

def cloneGlobals():
    ui = g.ui
    g.ui = None
    gCopy = copy.deepcopy(g)
    g.ui = ui
    return gCopy
    

