from Video import Video
from Align import Align
from Stack import Stack
from Globals import g
from pystackreg import StackReg

g.nThreads = 4
g.file = 'SATURN.MP4'
g.reference = '3'
g.driftP1 = (0, 0)
g.driftP2 = (0, 0)
g.transformation = StackReg.RIGID_BODY
g.limit = 100
g.blendMode = Stack.AVERAGE
g.sharpen1 = 0.00
g.sharpen2 = 0.00
g.sharpen3 = 0.00
g.radius1 = 0.50
g.radius2 = 0.50
g.radius3 = 0.50
g.denoise1 = 0.00
g.denoise2 = 0.00
g.denoise3 = 0.00
g.level1 = True
g.level2 = True
g.level3 = True

# Video
video = Video()
video.run()

# Align
align = Align(video.frames)
align.run()

# Stack
stack = Stack(align.similarities)
stack.run()
