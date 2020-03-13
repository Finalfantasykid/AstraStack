#!/usr/bin/python

import os
import sys
import time
import math
import cv2
from threading import Thread
from multiprocessing import Pipe
from pystackreg import StackReg

from Video import Video
from Align import Align
from Stack import Stack
from Sharpen import Sharpen
from Globals import g

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GObject, Pango

class UI:
    
    LOAD_TAB = 0
    ALIGN_TAB = 1
    STACK_TAB = 2
    SHARPEN_TAB = 3
    
    def __init__(self):
        self.parentConn, self.childConn = Pipe(duplex=False)
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
        self.tabs = self.builder.get_object("tabs")
        self.cpus = self.builder.get_object("cpus")
        self.progress = self.builder.get_object("progress")
        self.frame = self.builder.get_object("frame")
        self.frameSlider = self.builder.get_object("frameSlider")
        self.transformation = self.builder.get_object("transformation")
        self.limit = self.builder.get_object("limit")
        self.averageRadio = self.builder.get_object("averageRadio")
        self.medianRadio = self.builder.get_object("medianRadio")

        self.cpus.set_upper(os.cpu_count())
        self.cpus.set_value(math.ceil(os.cpu_count()/2))
        
        self.processThread = None
        
        self.builder.connect_signals(self)
        
        self.window.show_all()
        self.builder.get_object("blendGrid").hide() # Not overlly useful, might enable later
        self.setProgress()
        self.setTransformation()
        self.setThreads()
        
    # Sets up a listener so that threads can communicate with each other
    def createListener(self, conn, function):
        return GObject.io_add_watch(conn.fileno(), GObject.IO_IN, function)
        
    # Stops listening to the given listener
    def stopListening(self, watch):
        GObject.source_remove(watch)
    
    # Shows the error dialog with the given title and message
    def showErrorDialog(self, message):
        errorDialog = self.builder.get_object("errorDialog")
        errorDialog.format_secondary_text(message)
        response = errorDialog.run()
        errorDialog.hide()
        
    # Sets the number of threads to use
    def setThreads(self, *args):
        g.nThreads = int(self.cpus.get_value())
    
    # Opens the file chooser to open load a file
    def openFileDialog(self, *args):
        openDialog = self.builder.get_object("openDialog")
        response = openDialog.run()
        openDialog.hide()
        if(response == Gtk.ResponseType.OK):
            g.file = openDialog.get_filename()
            self.video = Video()
            thread = Thread(target=self.video.run, args=())
            thread.start()
            
    # Opens the file chooser to save the final image
    def saveFileDialog(self, *args):
        saveDialog = self.builder.get_object("saveDialog")
        response = saveDialog.run()
        saveDialog.hide()
        if(response == Gtk.ResponseType.OK):
            fileName = saveDialog.get_filename()
            try:
                cv2.imwrite(fileName, self.sharpen.finalImage)
            except: # Save Failed
                self.showErrorDialog("There was an error saving the image, make sure it is a valid file extension.")     
        
    # Called when the video is finished loading
    def finishedVideo(self):
        Gdk.threads_enter()
        self.tabs.next_page()
        self.limit.set_upper(len(self.video.frames))
        self.limit.set_value(int(len(self.video.frames)/2))
        g.driftP1 = (0, 0)
        g.driftP2 = (0, 0)

        self.setReference()
        self.setDriftPoint()
        self.setLimit()
        self.setBlendMode()
        Gdk.threads_leave()
        
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
            self.frame.set_from_file("sharpened.png")
        
    # Sets the reference frame to the current visible frame
    def setReference(self, *args):
        g.reference = str(int(self.frameSlider.get_value()))
        self.builder.get_object("referenceLabel").set_text(g.reference)
        
    # Updates the progress bar
    def setProgress(self, i=0, total=0, text=""):
        Gdk.threads_enter()
        if(total == 0):
            pass
            self.progress.hide()
        else:
            self.progress.show()
            self.progress.set_fraction(i/total)
            self.progress.set_text(text + " " + str(round((i/total)*100)) + "%")
        Gdk.threads_leave()
        
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
        
    # Called when the Alignment is complete
    def finishedAlign(self):
        Gdk.threads_enter()
        self.tabs.next_page()
        g.driftP1 = (0, 0)
        g.driftP2 = (0, 0)
        self.setDriftPoint()
        Gdk.threads_leave()
        
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
        
    # Called when the stack is complete
    def finishedStack(self):
        Gdk.threads_enter()
        self.sharpen = Sharpen("stacked.png")
        self.tabs.next_page()
        Gdk.threads_leave()
        
    # Sharpens the final Stacked image
    def sharpenImage(self, *args):
        g.sharpen1 = self.builder.get_object("sharpen1").get_value()
        g.sharpen2 = self.builder.get_object("sharpen2").get_value()
        g.sharpen3 = self.builder.get_object("sharpen3").get_value()

        g.radius1 = self.builder.get_object("radius1").get_value()
        g.radius2 = self.builder.get_object("radius2").get_value()
        g.radius3 = self.builder.get_object("radius3").get_value()
        
        g.denoise1 = self.builder.get_object("denoise1").get_value()
        g.denoise2 = self.builder.get_object("denoise2").get_value()
        g.denoise3 = self.builder.get_object("denoise3").get_value()
        
        g.level1 = self.builder.get_object("level1").get_active()
        g.level2 = self.builder.get_object("level2").get_active()
        g.level3 = self.builder.get_object("level3").get_active()
        
        if(self.sharpen is None):
            self.sharpen = Sharpen("stacked.png")
        if(self.processThread != None and self.processThread.is_alive()):
            self.sharpen.processAgain = True
        else:
            self.processThread = Thread(target=self.sharpen.run, args=())
            self.processThread.start()
            self.frame.set_from_file("sharpened.png")
        
    def finishedSharpening(self):
        Gdk.threads_enter()
        self.frame.set_from_file("sharpened.png")
        Gdk.threads_leave()

    # Closes the application
    def close(self, *args):
        Gtk.main_quit()

if __name__ == "__main__":
    g.ui = UI()
    GObject.threads_init()
    Gdk.threads_init()

    Gtk.main()
