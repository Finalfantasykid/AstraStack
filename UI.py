from os import cpu_count, rmdir, scandir, unlink, path
import math
import cv2
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
from gi.repository import Gtk, Gdk, GLib

class UI:
    
    LOAD_TAB = 0
    ALIGN_TAB = 1
    STACK_TAB = 2
    SHARPEN_TAB = 3
    
    def __init__(self):
        self.parentConn, self.childConn = Pipe(duplex=True)
        self.cleanTmp()
        self.video = None
        self.align = None
        self.stack = None
        self.sharpen = None
        self.mousePosition = None
        self.clickedDriftP1 = False
        self.clickedDriftP2 = False
        
        self.builder = Gtk.Builder()
        self.builder.add_from_file("ui/ui.glade")
        
        self.window = self.builder.get_object("mainWindow")
        self.saveDialog = self.builder.get_object("saveDialog")
        self.openDialog = self.builder.get_object("openDialog")
        self.tabs = self.builder.get_object("tabs")
        self.cpus = self.builder.get_object("cpus")
        self.progress = self.builder.get_object("progress")
        self.frame = self.builder.get_object("frame")
        self.frameSlider = self.builder.get_object("frameSlider")
        self.transformation = self.builder.get_object("transformation")
        self.limit = self.builder.get_object("limit")
        self.averageRadio = self.builder.get_object("averageRadio")
        self.medianRadio = self.builder.get_object("medianRadio")
        
        self.builder.get_object("alignTab").set_sensitive(False)
        self.builder.get_object("stackTab").set_sensitive(False)
        self.builder.get_object("processTab").set_sensitive(False)

        self.cpus.set_upper(cpu_count())
        self.cpus.set_value(math.ceil(cpu_count()/2))
        g.pool = ProcessPoolExecutor(max_workers=cpu_count())

        self.processThread = None
        
        self.builder.connect_signals(self)
        
        self.window.show_all()
        self.builder.get_object("blendGrid").hide() # Not overlly useful, might enable later
        self.setProgress()
        self.setTransformation()
        self.setThreads()
        
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
        errorDialog = self.builder.get_object("errorDialog")
        errorDialog.format_secondary_text(message)
        response = errorDialog.run()
        errorDialog.hide()
        
    # Disabled inputs
    def disableUI(self):
        self.builder.get_object("sidePanel").set_sensitive(False)
        
    def enableUI(self):
        self.builder.get_object("sidePanel").set_sensitive(True)
        
    # Sets the number of threads to use
    def setThreads(self, *args):
        g.nThreads = int(self.cpus.get_value())
    
    # Opens the file chooser to open load a file
    def openVideo(self, *args):
        self.openDialog.set_current_folder(path.expanduser("~"))
        self.openDialog.set_select_multiple(False)
        self.openDialog.set_filter(self.builder.get_object("videoFilter"))
        response = self.openDialog.run()
        self.openDialog.hide()
        if(response == Gtk.ResponseType.OK):
            g.file = self.openDialog.get_filename()
            self.video = Video()
            thread = Thread(target=self.video.run, args=())
            thread.start()
            self.disableUI()
            
    # Opens the file chooser to open load a file
    def openImageSequence(self, *args):
        self.openDialog.set_current_folder(path.expanduser("~"))
        self.openDialog.set_select_multiple(True)
        self.openDialog.set_filter(self.builder.get_object("imageFilter"))
        response = self.openDialog.run()
        self.openDialog.hide()
        if(response == Gtk.ResponseType.OK):
            g.file = self.openDialog.get_filenames()
            self.video = Video()
            thread = Thread(target=self.video.run, args=())
            thread.start()
            self.disableUI()
            
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
                cv2.imwrite(g.tmp + "stacked.png", cv2.imread(g.file))
                self.sharpen = Sharpen(g.tmp + "stacked.png", True)
                self.builder.get_object("alignTab").set_sensitive(False)
                self.builder.get_object("stackTab").set_sensitive(False)
                self.builder.get_object("processTab").set_sensitive(True)
                self.tabs.set_current_page(UI.SHARPEN_TAB)
                self.frame.set_from_file(g.tmp + "stacked.png")
            except: # Open Failed
                self.showErrorDialog("There was an error saving the image, make sure it is a valid file extension.")
            self.enableUI()
            
    # Opens the file chooser to save the final image
    def saveFileDialog(self, *args):
        self.saveDialog.set_current_folder(path.expanduser("~"))
        response = self.saveDialog.run()
        if(response == Gtk.ResponseType.OK):
            fileName = self.saveDialog.get_filename()
            try:
                cv2.imwrite(fileName, self.sharpen.finalImage)
            except: # Save Failed
                self.showErrorDialog("There was an error saving the image, make sure it is a valid file extension.")
        self.saveDialog.hide()
        
    # Called when the video is finished loading
    def finishedVideo(self):
        def update():
            self.tabs.next_page()
            self.limit.set_upper(len(self.video.frames))
            self.limit.set_value(int(len(self.video.frames)/2))
            g.driftP1 = (0, 0)
            g.driftP2 = (0, 0)

            self.setReference()
            self.setDriftPoint()
            self.setLimit()
            self.setBlendMode()
            self.enableUI()
            self.builder.get_object("alignTab").set_sensitive(True)
            self.builder.get_object("stackTab").set_sensitive(False)
            self.builder.get_object("processTab").set_sensitive(False)
            self.builder.get_object("alignButton").set_sensitive(False)
        GLib.idle_add(update)
        
    def changeTab(self, notebook, page, page_num, user_data=None):
        frameScale = self.builder.get_object("frameScale")
        if(page_num == UI.LOAD_TAB or page_num == UI.ALIGN_TAB):
            self.frameSlider.set_value(0)
            self.frameSlider.set_upper(len(self.video.frames)-1)
            frameScale.show()
            self.frame.set_from_file(self.video.frames[int(self.frameSlider.get_value())])
        elif(page_num == UI.STACK_TAB):
            self.frameSlider.set_value(0)
            self.frameSlider.set_upper(len(self.video.frames)-1)
            frameScale.show()
            self.frame.set_from_file(self.align.similarities[int(self.frameSlider.get_value())][0])
        elif(page_num == UI.SHARPEN_TAB):
            frameScale.hide()
            self.sharpenImage()
    
    # Changes the image frame to the frameSlider position    
    def updateImage(self, *args):
        page_num = self.tabs.get_current_page()
        if(page_num == UI.LOAD_TAB or page_num == UI.ALIGN_TAB):
            self.frame.set_from_file(self.video.frames[int(self.frameSlider.get_value())])
        elif(page_num == UI.STACK_TAB):
            self.frame.set_from_file(self.align.similarities[int(self.frameSlider.get_value())][0])
        elif(page_num == UI.SHARPEN_TAB):
            self.frame.set_from_file(g.tmp + "sharpened.png")
        
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
        
    # Drift Point 1 Button Clicked
    def clickDriftP1(self, *args):
        self.clickedDriftP1 = False
        self.clickedDriftP2 = False
        self.setDriftPoint()
        self.frameSlider.set_value(0)
        self.clickedDriftP1 = True
        self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.CROSSHAIR))
        
    # Drift Point 2 Button Clicked
    def clickDriftP2(self, *args):
        self.clickedDriftP1 = False
        self.clickedDriftP2 = False
        self.setDriftPoint()
        self.frameSlider.set_value(len(self.video.frames)-1)
        self.clickedDriftP2 = True
        self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.CROSSHAIR))
    
    # Updates the drift point
    def setDriftPoint(self, *args):
        if(self.clickedDriftP1 == False and self.clickedDriftP2 == False):
            # Just reset the label
            self.builder.get_object("driftPointLabel1").set_text(str(g.driftP1))
            self.builder.get_object("driftPointLabel2").set_text(str(g.driftP2))
        elif(self.clickedDriftP1):
            g.driftP1 = self.mousePosition
            self.builder.get_object("driftPointLabel1").set_text(str(g.driftP1))
        elif(self.clickedDriftP2):
            g.driftP2 = self.mousePosition
            self.builder.get_object("driftPointLabel2").set_text(str(g.driftP2))
        self.clickedDriftP1 = False
        self.clickedDriftP2 = False
        self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.LEFT_PTR))
    
    # Called when the mouse moves over the frame
    def updateMousePosition(self, *args):
        pointer = self.frame.get_pointer()
        self.mousePosition = (min(max(0, pointer.x), self.frame.get_allocation().width), min(max(0, pointer.y), self.frame.get_allocation().height))
        if(self.clickedDriftP1):
            self.builder.get_object("driftPointLabel1").set_text(str(self.mousePosition))
        if(self.clickedDriftP2):
            self.builder.get_object("driftPointLabel2").set_text(str(self.mousePosition))
    
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
        self.align = Align(self.video.frames)
        thread = Thread(target=self.align.run, args=())
        thread.start()
        self.disableUI()
        
    # Called when the Alignment is complete
    def finishedAlign(self):
        def update():
            self.tabs.next_page()
            g.driftP1 = (0, 0)
            g.driftP2 = (0, 0)
            self.setDriftPoint()
            self.enableUI()
            self.builder.get_object("alignTab").set_sensitive(True)
            self.builder.get_object("stackTab").set_sensitive(True)
            self.builder.get_object("processTab").set_sensitive(False)
        GLib.idle_add(update)
        
    # Sets the number of frames to use in the Stack
    def setLimit(self, *args):
        g.limit = int(self.limit.get_value())
       
    # Sets the blend mode of the Stack (Average or Median)
    def setBlendMode(self, *args):
        if(self.averageRadio.get_active()):
            g.blendMode = Stack.AVERAGE
        if(self.medianRadio.get_active()):
            g.blendMode = Stack.MEDIAN
            
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
        
        if(self.sharpen is None):
            if(self.stack is not None):
                self.sharpen = Sharpen(self.stack.stackedImage)
            else:
                self.sharpen = Sharpen(g.tmp + "stacked.png", True)
        if(self.processThread != None and self.processThread.is_alive()):
            self.sharpen.processAgain = True
        else:
            self.processThread = Thread(target=self.sharpen.run, args=())
            self.processThread.start()
            self.frame.set_from_file(g.tmp + "sharpened.png")
    
    # Called when sharpening is complete
    def finishedSharpening(self):
        def update():
            self.frame.set_from_file(g.tmp + "sharpened.png")
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
    g.ui = UI()
    
    Gtk.Settings.get_default().set_property("gtk-theme-name", "Adwaita")
    Gtk.Settings.get_default().set_property("gtk-icon-theme-name", "Adwaita")

    Gtk.main()

