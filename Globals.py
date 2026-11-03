import copy

class Object():
    def actualColor(self):
        return (self.colorMode or self.guessedColorMode)

g = Object()

g.TESTING = False
g.ui = None

def cloneGlobals():
    ui = g.ui
    g.ui = None
    gCopy = copy.deepcopy(g)
    g.ui = ui
    return gCopy
    

