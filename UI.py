from os import cpu_count, rmdir, scandir, unlink, path
import math
import cv2
import webbrowser
import urllib.request
import ssl
import json
import math
import psutil
from threading import Thread
from multiprocessing import Pipe
from concurrent.futures import ProcessPoolExecutor
from pystackreg import StackReg

from Video import Video
from Align import Align
from Stack import Stack
from Sharpen import Sharpen
from Globals import g

from gi import require_version
require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

ssl._create_default_https_context = ssl._create_unverified_context

class UI:
    
    LOAD_TAB = 0
    ALIGN_TAB = 1
    STACK_TAB = 2
    SHARPEN_TAB = 3
    
    TITLE = "AstraStack"
    VERSION = "1.3.0"
    
    def __init__(self):
        self.parentConn, self.childConn = Pipe(duplex=True)
        self.cleanTmp()
        self.newVersionUrl = ""
        self.video = None
        self.align = None
        self.stack = None
        self.sharpen = None
        self.mousePosition = None
        self.clickedDriftP1 = False
        self.clickedDriftP2 = False
        self.clickedProcessingArea = False
        
        self.builder = Gtk.Builder()
        self.builder.add_from_file("ui/ui.glade")
        
        self.window = self.builder.get_object("mainWindow")
        self.saveDialog = self.builder.get_object("saveDialog")
        self.openDialog = self.builder.get_object("openDialog")
        self.tabs = self.builder.get_object("tabs")
        self.cpus = self.builder.get_object("cpus")
        self.progress = self.builder.get_object("progress")
        self.frame = self.builder.get_object("frame")
        self.overlay = self.builder.get_object("overlay")
        self.frameSlider = self.builder.get_object("frameSlider")
        self.frameScale = self.builder.get_object("frameScale")
        self.startFrame = self.builder.get_object("startFrame")
        self.endFrame = self.builder.get_object("endFrame")
        self.normalize = self.builder.get_object("normalize")
        self.alignChannels = self.builder.get_object("alignChannels")
        self.transformation = self.builder.get_object("transformation")
        self.limit = self.builder.get_object("limit")
        self.limitPercent = self.builder.get_object("limitPercent")
        self.averageRadio = self.builder.get_object("averageRadio")
        self.medianRadio = self.builder.get_object("medianRadio")
        
        self.builder.get_object("alignTab").set_sensitive(False)
        self.builder.get_object("stackTab").set_sensitive(False)
        self.builder.get_object("processTab").set_sensitive(False)
        
        self.builder.get_object("blackLevel").add_mark(0, Gtk.PositionType.TOP, None)
        self.builder.get_object("gamma").add_mark(100, Gtk.PositionType.TOP, None)
        self.builder.get_object("value").add_mark(100, Gtk.PositionType.TOP, None)
        self.builder.get_object("red").add_mark(255, Gtk.PositionType.TOP, None)
        self.builder.get_object("green").add_mark(255, Gtk.PositionType.TOP, None)
        self.builder.get_object("blue").add_mark(255, Gtk.PositionType.TOP, None)
        self.builder.get_object("saturation").add_mark(100, Gtk.PositionType.TOP, None)

        self.cpus.set_upper(min(61, cpu_count())) # 61 is the maximum that Windows allows
        self.cpus.set_value(min(61, math.ceil(cpu_count()/2)))
        g.pool = None

        self.processThread = None
        
        self.builder.connect_signals(self)
        
        # Needed so it can be temporarily removed
        self.limitPercentSignal = self.limitPercent.connect("value-changed", self.setLimitPercent) 
        
        g.driftP1 = (0, 0)
        g.driftP2 = (0, 0)
        
        g.processingAreaP1 = (0, 0)
        g.processingAreaP2 = (0, 0)
        
        self.window.show_all()
        self.checkNewVersion()
        self.setProgress()
        self.setNormalize()
        self.setAlignChannels()
        self.setTransformation()
        self.setThreads()
        self.frameScale.set_sensitive(False)
        g.reference = "0"
        
    def propagateScroll(self, widget, event):
        Gtk.propagate_event(widget.get_parent(), event)
    
    # Sets up a listener so that processes can communicate with each other
    def createListener(self, function):
        def listener(function):
            while True:
                try:
                    msg = self.parentConn.recv()
                except:
                    return False
                if(msg == "stop"):
                    return False
                function(msg)
                
        thread = Thread(target=listener, args=(function,))
        thread.start()
        return thread
    
    # Shows the error dialog with the given title and message
    def showErrorDialog(self, message):
        dialog = self.builder.get_object("errorDialog")
        dialog.format_secondary_text(message)
        response = dialog.run()
        dialog.hide()
        return response
        
    # Shows the warning dialog with the given title and message
    def showWarningDialog(self, message):
        dialog = self.builder.get_object("warningDialog")
        dialog.format_secondary_text(message)
        response = dialog.run()
        dialog.hide()
        return response
        
    # Opens the About dialog
    def showAbout(self, *args):
        dialog = self.builder.get_object("about")
        dialog.set_program_name(UI.TITLE)
        dialog.set_version(UI.VERSION)
        response = dialog.run()
        dialog.hide()
        
    # Disable inputs
    def disableUI(self):
        self.builder.get_object("sidePanel").set_sensitive(False)
        self.builder.get_object("toolbar").set_sensitive(False)
        
    # Enable inputs
    def enableUI(self):
        self.builder.get_object("sidePanel").set_sensitive(True)
        self.builder.get_object("toolbar").set_sensitive(True)
        
    # The following is needed to forcibly refresh the value spacing of the slider
    def fixFrameSliderBug(self):
        self.frameScale.set_value_pos(Gtk.PositionType.RIGHT)
        self.frameScale.set_value_pos(Gtk.PositionType.LEFT)
        
    # Sets the number of threads to use
    def setThreads(self, *args):
        g.nThreads = int(self.cpus.get_value())
        if(g.pool is not None):
            g.pool.shutdown()
        g.pool = ProcessPoolExecutor(g.nThreads)
        
    def checkNewVersion(self):
        def callUrl():
            try:
                contents = urllib.request.urlopen("https://api.github.com/repos/Finalfantasykid/AstraStack/releases/latest").read()
                obj = json.loads(contents)
                if(obj['name'] > UI.VERSION):
                    self.newVersionUrl = obj['html_url']
                    button.show()
            except:
                return
        button = self.builder.get_object("newVersion")
        button.hide()
        thread = Thread(target=callUrl, args=())
        thread.start()
        
    # Opens the GitHub releases page in a browser
    def clickNewVersion(self, *args):
        webbrowser.open(self.newVersionUrl)
    
    # Checks to see if there will be enough memory to process the image
    def checkMemory(self, w=0, h=0):
        if(Sharpen.estimateMemoryUsage(w, h) > psutil.virtual_memory().available):
            response = self.showWarningDialog("Your system may not have enough memory to process this file, are you sure you want to continue?")
            return (response == Gtk.ResponseType.YES)
        return True
    
    # Opens the file chooser to open load a file
    def openVideo(self, *args):
        self.openDialog.set_current_folder(path.expanduser("~"))
        self.openDialog.set_select_multiple(False)
        self.openDialog.set_filter(self.builder.get_object("videoFilter"))
        response = self.openDialog.run()
        self.openDialog.hide()
        if(response == Gtk.ResponseType.OK):
            try:
                g.file = self.openDialog.get_filename()
                self.video = Video()
                self.video.checkMemory()
                thread = Thread(target=self.video.run, args=())
                thread.start()
                self.disableUI()
            except MemoryError as error:
                self.enableUI()
            
    # Opens the file chooser to open load a file
    def openImageSequence(self, *args):
        self.openDialog.set_current_folder(path.expanduser("~"))
        self.openDialog.set_select_multiple(True)
        self.openDialog.set_filter(self.builder.get_object("imageFilter"))
        response = self.openDialog.run()
        self.openDialog.hide()
        if(response == Gtk.ResponseType.OK):
            try:
                g.file = self.openDialog.get_filenames()
                self.video = Video()
                self.video.checkMemory()
                thread = Thread(target=self.video.run, args=())
                thread.start()
                self.disableUI()
            except MemoryError as error:
                self.enableUI()
            
    # Opens the file chooser to open load a file
    def openImage(self, *args):
        self.openDialog.set_current_folder(path.expanduser("~"))
        self.openDialog.set_select_multiple(False)
        self.openDialog.set_filter(self.builder.get_object("imageFilter"))
        response = self.openDialog.run()
        self.openDialog.hide()
        if(response == Gtk.ResponseType.OK):
            self.disableUI()
            g.file = self.openDialog.get_filename()
            try:
                self.video = Video()
                self.video.mkdirs()
                img = cv2.imread(g.file)
                h, w = img.shape[:2]
                if(not self.checkMemory(w, h)):
                    raise MemoryError()
                cv2.imwrite(g.tmp + "stacked.png", img)
                
                self.window.set_title(path.split(g.file)[1] + " - " + UI.TITLE)
                self.sharpen = Sharpen(g.tmp + "stacked.png", True)
                self.builder.get_object("alignTab").set_sensitive(False)
                self.builder.get_object("stackTab").set_sensitive(False)
                self.builder.get_object("processTab").set_sensitive(True)
                self.tabs.set_current_page(UI.SHARPEN_TAB)
                self.frame.set_from_file(g.tmp + "stacked.png")
            except MemoryError as error:
                pass
            except: # Open Failed
                self.showErrorDialog("There was an error opening the image, make sure it is a valid image.")
            self.enableUI()
            
    # Opens the file chooser to save the final image
    def saveFileDialog(self, *args):
        self.saveDialog.set_current_folder(path.expanduser("~"))
        response = self.saveDialog.run()
        if(response == Gtk.ResponseType.OK):
            fileName = self.saveDialog.get_filename()
            try:
                cv2.imwrite(fileName, cv2.cvtColor(self.sharpen.finalImage.astype('uint8'), cv2.COLOR_RGB2BGR))
            except: # Save Failed
                self.showErrorDialog("There was an error saving the image, make sure it is a valid file extension.")
        self.saveDialog.hide()
        
    # Called when the video is finished loading
    def finishedVideo(self):
        def update():
            self.tabs.next_page()
           
            self.frameScale.set_sensitive(True)
           
            self.startFrame.set_lower(0)
            self.startFrame.set_upper(len(self.video.frames)-1)
            self.startFrame.set_value(0)
            
            self.endFrame.set_upper(len(self.video.frames)-1)
            self.endFrame.set_lower(0)
            self.endFrame.set_value(len(self.video.frames)-1)
            
            g.driftP1 = (0, 0)
            g.driftP2 = (0, 0)
            g.processingAreaP1 = (0, 0)
            g.processingAreaP2 = (0, 0)
            g.reference = self.video.sharpest
            self.frameSlider.set_value(self.video.sharpest)

            self.setReference()
            self.setStartFrame()
            self.setEndFrame()
            self.setDriftPoint()
            self.enableUI()
            if(isinstance(g.file, list)):
                sList = sorted(g.file)
                self.window.set_title(path.split(sList[0])[1] + " ... " + path.split(sList[-1])[1] +  " - " + UI.TITLE)
            else:
                self.window.set_title(path.split(g.file)[1] + " - " + UI.TITLE)
            self.builder.get_object("alignTab").set_sensitive(True)
            self.builder.get_object("stackTab").set_sensitive(False)
            self.builder.get_object("processTab").set_sensitive(False)
        GLib.idle_add(update)
        
    # Called when the tab is changed.  Updates parts of the UI based on the tab
    def changeTab(self, notebook, page, page_num, user_data=None):
        if(page_num == UI.LOAD_TAB or page_num == UI.ALIGN_TAB):
            self.frameSlider.set_value(0)
            self.frameSlider.set_upper(len(self.video.frames)-1)
            self.setStartFrame()
            self.setEndFrame()
            self.frameScale.show()
            self.frame.set_from_file(self.video.frames[int(self.frameSlider.get_value())])
        elif(page_num == UI.STACK_TAB):
            self.frameSlider.set_lower(0)
            self.frameSlider.set_upper(len(self.align.similarities)-1)
            self.frameSlider.set_value(0)
            self.frameScale.show()
            self.frame.set_from_file(self.align.similarities[int(self.frameSlider.get_value())][0])
        elif(page_num == UI.SHARPEN_TAB):
            self.frameScale.hide()
            self.sharpenImage()
        self.fixFrameSliderBug()
    
    # Changes the image frame to the frameSlider position    
    def updateImage(self, *args):
        page_num = self.tabs.get_current_page()
        if(page_num == UI.LOAD_TAB or page_num == UI.ALIGN_TAB):
            self.frame.set_from_file(self.video.frames[int(self.frameSlider.get_value())])
        elif(page_num == UI.STACK_TAB):
            self.frame.set_from_file(self.align.similarities[int(self.frameSlider.get_value())][0])
            
    def drawOverlay(self, widget, cr):
        width = widget.get_allocated_width()
        height = widget.get_allocated_height()
        
        def drawPoint(cr, x, y):
            cr.new_sub_path()
            cr.set_line_width(2)
            cr.set_source_rgb(1, 1, 1)

            cr.arc(x, y, 2, 0, 2*math.pi)
            cr.stroke_preserve()
            
            cr.set_source_rgb(1, 0, 0)
            cr.fill()
            
        def drawRect(cr, x1, y1, x2, y2):
            cr.rectangle(0, 0, x1, y1)
            cr.rectangle(0, y1, x1, (y2-y1))
            cr.rectangle(0, y1, x1, height*2)
            cr.rectangle(x1, y2, (x2-x1), height*2)
            cr.rectangle(x2, y2, width*2, height*2)
            cr.rectangle(x2, y1, width*2, (y2-y1))
            cr.rectangle(x2, 0, width*2, y1)
            cr.rectangle(x1, 0, (x2-x1), y1)
            
            cr.set_source_rgba(0, 0, 0, 0.25)
            cr.fill()
            
            cr.set_line_width(1)
            cr.set_source_rgb(1, 0, 0)
            
            cr.rectangle(x1, y1, (x2-x1), (y2-y1))
            cr.stroke()
            
        if(self.tabs.get_current_page() == UI.ALIGN_TAB):
            current = self.frameSlider.get_value()
            
            # Processing Area
            px1 = min(g.processingAreaP1[0], g.processingAreaP2[0])
            py1 = min(g.processingAreaP1[1], g.processingAreaP2[1])
            
            px2 = max(g.processingAreaP1[0], g.processingAreaP2[0])
            py2 = max(g.processingAreaP1[1], g.processingAreaP2[1])
            
            # Drift Points
            dx1 = g.driftP1[0]
            dy1 = g.driftP1[1]
            
            dx2 = g.driftP2[0]
            dy2 = g.driftP2[1]
            
            dx = 0
            dy = 0
            
            if(dx1 != 0 and dy1 != 0 and
               dx2 != 0 and dy2 != 0):
                dx = dx2 - dx1
                dy = dy2 - dy1
            
            if(px1 != 0 and py1 != 0 and
               px2 != 0 and py2 != 0):
                # Draw Processing Area Rectangle
                drawRect(cr, px1 + (dx/(g.endFrame - g.startFrame))*(current-g.startFrame), 
                             py1 + (dy/(g.endFrame - g.startFrame))*(current-g.startFrame), 
                             px2 + (dx/(g.endFrame - g.startFrame))*(current-g.startFrame), 
                             py2 + (dy/(g.endFrame - g.startFrame))*(current-g.startFrame))
                
            if(dx1 != 0 and dy1 != 0 and current == g.startFrame):
                # Draw point on first frame
                drawPoint(cr, dx1, dy1)
                
            if(dx1 != 0 and dy1 != 0 and current != g.startFrame and
               dx2 != 0 and dy2 != 0 and current != g.endFrame):
                # Draw interpolated point
                drawPoint(cr, dx1 + (dx/(g.endFrame - g.startFrame))*(current-g.startFrame), 
                              dy1 + (dy/(g.endFrame - g.startFrame))*(current-g.startFrame))
                
            if(dx2 != 0 and dy2 != 0 and current == g.endFrame):
                # Draw point on last frame
                drawPoint(cr, dx2, dy2)
        
    # Sets the reference frame to the current visible frame
    def setReference(self, *args):
        g.reference = str(int(self.frameSlider.get_value()))
        self.builder.get_object("referenceLabel").set_text(g.reference)
        self.builder.get_object("alignButton").set_sensitive(True)
        
    # Updates the progress bar
    def setProgress(self, i=0, total=0, text=""):
        def update():
            if(total == 0):
                pass
                self.progress.hide()
            else:
                self.progress.show()
                self.progress.set_fraction(i/total)
                self.progress.set_text(text + " " + str(round((i/total)*100)) + "%")
        GLib.idle_add(update)
        
    # Sets the start frame for trimming
    def setStartFrame(self, *args):
        g.startFrame = int(self.startFrame.get_value())
        self.endFrame.set_lower(g.startFrame+1)
        self.frameSlider.set_lower(g.startFrame)
        self.frameSlider.set_value(max(g.startFrame, self.frameSlider.get_value()))
        if(int(g.startFrame) > int(g.reference)):
            # Reference is outside of the range, fix it
            g.reference = str(int(g.startFrame))
            self.builder.get_object("referenceLabel").set_text(g.reference)
        self.fixFrameSliderBug()
        
    # Sets the end frame for trimming
    def setEndFrame(self, *args):
        g.endFrame = int(self.endFrame.get_value())
        self.startFrame.set_upper(g.endFrame-1)
        self.frameSlider.set_upper(g.endFrame)
        self.frameSlider.set_value(min(g.endFrame, self.frameSlider.get_value()))
        if(int(g.endFrame) < int(g.reference)):
            # Reference is outside of the range, fix it
            g.reference = str(int(g.endFrame))
            self.builder.get_object("referenceLabel").set_text(g.reference)
        self.fixFrameSliderBug()
        
    # Drift Point 1 Button Clicked
    def clickDriftP1(self, *args):
        self.clickedDriftP1 = False
        self.clickedDriftP2 = False
        self.clickedProcessingArea = False
        self.setDriftPoint()
        self.frameSlider.set_value(g.startFrame)
        self.clickedDriftP1 = True
        self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.CROSSHAIR))
        
    # Drift Point 2 Button Clicked
    def clickDriftP2(self, *args):
        self.clickedDriftP1 = False
        self.clickedDriftP2 = False
        self.clickedProcessingArea = False
        self.setDriftPoint()
        self.frameSlider.set_value(g.endFrame)
        self.clickedDriftP2 = True
        self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.CROSSHAIR))
        
    # Reset Drift Point 1 to (0, 0)
    def resetDriftP1(self, widget, event):
        if(event.button == 3): # Right Click
            g.driftP1 = (0, 0)
            self.clickedDriftP1 = False
            self.clickedDriftP2 = False
            self.clickedProcessingArea = False
            self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.LEFT_PTR))
            self.overlay.queue_draw()
            
    # Reset Drift Point 2 to (0, 0)
    def resetDriftP2(self, widget, event):
        if(event.button == 3): # Right Click
            g.driftP2 = (0, 0)
            self.clickedDriftP1 = False
            self.clickedDriftP2 = False
            self.clickedProcessingArea = False
            self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.LEFT_PTR))
            self.overlay.queue_draw()
    
    # Updates the drift point
    def setDriftPoint(self, *args):
        if(self.clickedDriftP1 or self.clickedDriftP2):
            if(self.clickedDriftP1):
                g.driftP1 = self.mousePosition
            elif(self.clickedDriftP2):
                g.driftP2 = self.mousePosition
            self.clickedDriftP1 = False
            self.clickedDriftP2 = False
            self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.LEFT_PTR))
            self.overlay.queue_draw()
        
    def clickProcessingArea(self, *args):
        g.processingAreaP1 = (0, 0)
        g.processingAreaP2 = (0, 0)
        self.clickedDriftP1 = False
        self.clickedDriftP2 = False
        self.clickedProcessingArea = True
        self.frameSlider.set_value(g.startFrame)
        self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.CROSSHAIR))
        self.overlay.queue_draw()
        
    # Reset Processing Area to (0, 0)
    def resetProcessingArea(self, widget, event):
        if(event.button == 3): # Right Click
            g.processingAreaP1 = (0, 0)
            g.processingAreaP2 = (0, 0)
            self.clickedDriftP1 = False
            self.clickedDriftP2 = False
            self.clickedProcessingArea = False
            self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.LEFT_PTR))
            self.overlay.queue_draw()
        
    def dragBegin(self, *args):
        if(self.clickedProcessingArea):
            g.processingAreaP1 = self.mousePosition
        
    def dragEnd(self, *args):
        if(self.clickedProcessingArea):
            g.processingAreaP2 = self.mousePosition
            self.clickedProcessingArea = False
            self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.LEFT_PTR))
    
    # Called when the mouse moves over the frame
    def updateMousePosition(self, *args):
        pointer = self.frame.get_pointer()
        self.mousePosition = (min(max(0, pointer.x), self.frame.get_allocation().width), min(max(0, pointer.y), self.frame.get_allocation().height))
        if(self.clickedProcessingArea):
            if(g.processingAreaP1 != (0, 0)):
                g.processingAreaP2 = self.mousePosition
            self.overlay.queue_draw()
    
    # Sets whether or not to normalize the frames during alignment
    def setNormalize(self, *args):
        g.normalize = self.normalize.get_active()
        
    # Sets whether or not to align channels separately
    def setAlignChannels(self, *args):
        g.alignChannels = self.alignChannels.get_active()
    
    # Sets the type of transformation
    def setTransformation(self, *args):
        text = self.transformation.get_active_text()
        if(text == "Translation"):
            g.transformation = StackReg.TRANSLATION
        elif(text == "Rigid Body"):
            g.transformation = StackReg.RIGID_BODY
        elif(text == "Scaled Rotation"):
            g.transformation = StackReg.SCALED_ROTATION
        elif(text == "Affine"):
            g.transformation = StackReg.AFFINE
            
    # Runs the Alignment step
    def clickAlign(self, *args):
        self.align = Align(self.video.frames[g.startFrame:g.endFrame+1])
        thread = Thread(target=self.align.run, args=())
        thread.start()
        self.disableUI()
        
    # Called when the Alignment is complete
    def finishedAlign(self):
        def update():
            self.tabs.next_page()
            self.enableUI()
            self.builder.get_object("alignTab").set_sensitive(True)
            self.builder.get_object("stackTab").set_sensitive(True)
            self.builder.get_object("processTab").set_sensitive(False)
            self.limit.set_upper(len(self.align.similarities))
            self.limit.set_value(int(len(self.align.similarities)/2))
            self.limitPercent.set_value(round(self.limit.get_value()/len(self.align.similarities)*100))
            self.setLimit()
            self.setLimitPercent()
        GLib.idle_add(update)
        
    # Sets the number of frames to use in the Stack
    def setLimit(self, *args):
        self.limitPercent.disconnect(self.limitPercentSignal)
        self.limit.set_upper(len(self.align.similarities))
        g.limit = int(self.limit.get_value())
        self.limitPercent.set_value(round(g.limit/len(self.align.similarities)*100))
        self.limitPercentSignal = self.limitPercent.connect("value-changed", self.setLimitPercent)
        
    # Sets the number of frames to use in the Stack
    def setLimitPercent(self, *args):
        limitPercent = self.limitPercent.get_value()/100
        self.limit.set_value(round(limitPercent*len(self.align.similarities)))
            
    # Stack Button clicked
    def clickStack(self, *args):
        self.stack = Stack(self.align.similarities)
        thread = Thread(target=self.stack.run, args=())
        thread.start()
        self.disableUI()
        
    # Called when the stack is complete
    def finishedStack(self):
        def update():
            self.sharpen = Sharpen(self.stack.stackedImage)
            self.tabs.next_page()
            self.frame.set_from_file(g.tmp + "stacked.png")
            self.enableUI()
            self.builder.get_object("alignTab").set_sensitive(True)
            self.builder.get_object("stackTab").set_sensitive(True)
            self.builder.get_object("processTab").set_sensitive(True)
        GLib.idle_add(update)
        
    # Sharpens the final Stacked image
    def sharpenImage(self, *args):
        g.sharpen1 = self.builder.get_object("sharpen1").get_value()
        g.sharpen2 = self.builder.get_object("sharpen2").get_value()
        g.sharpen3 = self.builder.get_object("sharpen3").get_value()
        g.sharpen4 = self.builder.get_object("sharpen4").get_value()
        g.sharpen5 = self.builder.get_object("sharpen5").get_value()

        g.radius1 = self.builder.get_object("radius1").get_value()
        g.radius2 = self.builder.get_object("radius2").get_value()
        g.radius3 = self.builder.get_object("radius3").get_value()
        g.radius4 = self.builder.get_object("radius4").get_value()
        g.radius5 = self.builder.get_object("radius5").get_value()
        
        g.denoise1 = self.builder.get_object("denoise1").get_value()
        g.denoise2 = self.builder.get_object("denoise2").get_value()
        g.denoise3 = self.builder.get_object("denoise3").get_value()
        g.denoise4 = self.builder.get_object("denoise4").get_value()
        g.denoise5 = self.builder.get_object("denoise5").get_value()
        
        g.level1 = self.builder.get_object("level1").get_active()
        g.level2 = self.builder.get_object("level2").get_active()
        g.level3 = self.builder.get_object("level3").get_active()
        g.level4 = self.builder.get_object("level4").get_active()
        g.level5 = self.builder.get_object("level5").get_active()
        
        g.gamma = self.builder.get_object("gammaAdjust").get_value()
        g.blackLevel = self.builder.get_object("blackLevelAdjust").get_value()
        g.value = self.builder.get_object("valueAdjust").get_value()
        
        g.redAdjust = self.builder.get_object("redAdjust").get_value()
        g.greenAdjust = self.builder.get_object("greenAdjust").get_value()
        g.blueAdjust = self.builder.get_object("blueAdjust").get_value()
        g.saturation = self.builder.get_object("saturationAdjust").get_value()
        
        if(len(args) > 0 and (self.builder.get_object("gammaAdjust") == args[0] or
                              self.builder.get_object("blackLevelAdjust") == args[0] or
                              self.builder.get_object("redAdjust") == args[0] or
                              self.builder.get_object("greenAdjust") == args[0] or
                              self.builder.get_object("blueAdjust") == args[0] or
                              self.builder.get_object("saturationAdjust") == args[0] or
                              self.builder.get_object("valueAdjust") == args[0])):
            processAgain = self.sharpen.processAgain
            processColor = True
        else:
            processAgain = True
            processColor = False
        
        if(self.sharpen is None):
            if(self.stack is not None):
                self.sharpen = Sharpen(self.stack.stackedImage)
            else:
                self.sharpen = Sharpen(g.tmp + "stacked.png", True)
        if(self.processThread != None and self.processThread.is_alive()):
            self.sharpen.processAgain = processAgain
            self.sharpen.processColorAgain = processColor
        else:
            self.processThread = Thread(target=self.sharpen.run, args=(processAgain, processColor))
            self.processThread.start()
    
    # Called when sharpening is complete
    def finishedSharpening(self):
        def update():
            z = self.sharpen.finalImage.astype('uint8').tobytes()
            Z = GLib.Bytes.new(z)
            pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(Z, GdkPixbuf.Colorspace.RGB, False, 8, self.sharpen.w, self.sharpen.h, self.sharpen.w*3)
            self.frame.set_from_pixbuf(pixbuf)
        GLib.idle_add(update)

    # Cleans the tmp directory
    def cleanTmp(self):
        if(path.exists(g.tmp)):
            if(path.exists(g.tmp + "frames")):
                for file in scandir(g.tmp + "frames"):
                    unlink(file.path)
            if(path.exists(g.tmp + "cache")):
                for file in scandir(g.tmp + "cache"):
                    unlink(file.path)
            for file in scandir(g.tmp):
                if(path.isdir(file)):
                    rmdir(file.path)
                else:
                    unlink(file.path)
            rmdir(g.tmp)

    # Closes the application
    def close(self, *args):
        Gtk.main_quit()
        self.cleanTmp()

def run():
    # Newer versions of Adwaita scalable icons don't work well with older librsvg.
    # This can be removed when no longer being built with an older librsvg
    Gtk.IconTheme.get_default().prepend_search_path('share_override/icons')

    g.ui = UI()

    Gtk.main()

