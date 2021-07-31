from os import cpu_count, rmdir, scandir, unlink, path
import subprocess, os, sys
from pathlib import Path
import math
import cv2
import numpy as np
import webbrowser
import urllib.request
import ssl
import json
import math
import psutil
import time
from packaging import version
from threading import Thread
from multiprocessing import active_children, get_start_method
from concurrent.futures import ProcessPoolExecutor
from pystackreg import StackReg

from deconvolution import *
from Video import Video
from Align import Align
from Stack import Stack, transform
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
    VERSION = "2.1.0"
    
    def __init__(self):
        self.pids = []
        self.newVersionUrl = ""
        self.video = None
        self.align = None
        self.stack = None
        self.sharpen = None
        self.mousePosition = None
        self.clickedDriftP1 = False
        self.clickedDriftP2 = False
        self.clickedAreaOfInterest = False
        
        self.builder = Gtk.Builder()
        self.builder.add_from_file("ui/ui.glade")
        
        self.window = self.builder.get_object("mainWindow")
        self.saveDialog = self.builder.get_object("saveDialog")
        self.openDialog = self.builder.get_object("openDialog")
        self.tabs = self.builder.get_object("tabs")
        self.cpus = self.builder.get_object("cpus")
        self.colorMode = self.builder.get_object("colorMode")
        self.progressBox = self.builder.get_object("progressBox")
        self.progress = self.builder.get_object("progress")
        self.processSpinner = self.builder.get_object("processSpinner")
        self.frame = self.builder.get_object("frame")
        self.overlay = self.builder.get_object("overlay")
        self.psfImage = self.builder.get_object("psfImage")
        self.frameSlider = self.builder.get_object("frameSlider")
        self.frameScale = self.builder.get_object("frameScale")
        self.startFrame = self.builder.get_object("startFrame")
        self.endFrame = self.builder.get_object("endFrame")
        self.normalize = self.builder.get_object("normalize")
        self.alignChannels = self.builder.get_object("alignChannels")
        self.autoCrop = self.builder.get_object("autoCrop")
        self.transformation = self.builder.get_object("transformation")
        self.bitDepth = self.builder.get_object("bitDepth")
        self.drizzleFactor = self.builder.get_object("drizzleFactor")
        self.drizzleInterpolation = self.builder.get_object("drizzleInterpolation")
        self.limit = self.builder.get_object("limit")
        self.limitPercent = self.builder.get_object("limitPercent")
        self.averageRadio = self.builder.get_object("averageRadio")
        self.medianRadio = self.builder.get_object("medianRadio")
        
        self.openDialog.set_preview_widget(Gtk.Image())
        self.saveDialog.set_preview_widget(Gtk.Image())
        
        self.builder.get_object("alignTab").set_sensitive(False)
        self.builder.get_object("stackTab").set_sensitive(False)
        self.builder.get_object("processTab").set_sensitive(False)
        
        self.builder.get_object("deconvolveCircularDiameterWidget").add_mark(1, Gtk.PositionType.TOP, None)
        self.builder.get_object("deconvolveCircularAmountWidget").add_mark(25, Gtk.PositionType.TOP, None)
        self.builder.get_object("deconvolveGaussianDiameterWidget").add_mark(1, Gtk.PositionType.TOP, None)
        self.builder.get_object("deconvolveGaussianAmountWidget").add_mark(25, Gtk.PositionType.TOP, None)
        self.builder.get_object("deconvolveLinearDiameterWidget").add_mark(1, Gtk.PositionType.TOP, None)
        self.builder.get_object("deconvolveLinearAmountWidget").add_mark(25, Gtk.PositionType.TOP, None)
        self.builder.get_object("deconvolveLinearAngleWidget").add_mark(0, Gtk.PositionType.TOP, None)
        self.builder.get_object("deconvolveCustomAmountWidget").add_mark(25, Gtk.PositionType.TOP, None)
        self.builder.get_object("blackLevel").add_mark(0, Gtk.PositionType.TOP, None)
        self.builder.get_object("gamma").add_mark(100, Gtk.PositionType.TOP, None)
        self.builder.get_object("value").add_mark(100, Gtk.PositionType.TOP, None)
        self.builder.get_object("red").add_mark(100, Gtk.PositionType.TOP, None)
        self.builder.get_object("green").add_mark(100, Gtk.PositionType.TOP, None)
        self.builder.get_object("blue").add_mark(100, Gtk.PositionType.TOP, None)
        self.builder.get_object("saturation").add_mark(100, Gtk.PositionType.TOP, None)

        self.disableScroll()

        self.cpus.set_upper(min(61, cpu_count())) # 61 is the maximum that Windows allows
        self.cpus.set_value(min(61, math.ceil(cpu_count()/2)))
        g.pool = None

        self.processThread = None
        
        self.builder.connect_signals(self)
        
        # Needed so it can be temporarily removed
        self.limitPercentSignal = self.limitPercent.connect("value-changed", self.setLimitPercent)
        
        # Default Open/Save Dialog buttons
        self.openDialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        self.openDialog.add_button("Open", Gtk.ResponseType.OK)
        self.saveDialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        self.saveDialog.add_button("Save", Gtk.ResponseType.OK)
        
        g.driftP1 = (0, 0)
        g.driftP2 = (0, 0)
        
        g.areaOfInterestP1 = (0, 0)
        g.areaOfInterestP2 = (0, 0)
        
        g.guessedColorMode = Video.COLOR_RGB
        
        self.window.show_all()
        self.checkNewVersion()
        self.setProgress()
        self.setColorMode()
        self.setNormalize()
        self.setAlignChannels()
        self.setTransformation()
        self.setDrizzleFactor()
        self.setDrizzleInterpolation()
        self.setAutoCrop()
        self.setThreads()
        self.frameScale.set_sensitive(False)
        
        g.file = None
        g.reference = "0"
        g.deconvolveCustomFile = None
        
    # Cancels scroll event for widget
    def propagateScroll(self, widget, event):
        Gtk.propagate_event(widget.get_parent(), event)
        
    # Disables the scroll event from some fields
    def disableScroll(self):
        mask = Gdk.EventMask.BUTTON_MOTION_MASK | Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK | Gdk.EventMask.KEY_RELEASE_MASK | Gdk.EventMask.TOUCH_MASK
        self.builder.get_object("denoiseWidget1").set_events(mask)
        self.builder.get_object("denoiseWidget2").set_events(mask)
        self.builder.get_object("denoiseWidget3").set_events(mask)
        self.builder.get_object("denoiseWidget4").set_events(mask)
        self.builder.get_object("denoiseWidget5").set_events(mask)
        
        self.builder.get_object("radiusWidget1").set_events(mask)
        self.builder.get_object("radiusWidget2").set_events(mask)
        self.builder.get_object("radiusWidget3").set_events(mask)
        self.builder.get_object("radiusWidget4").set_events(mask)
        self.builder.get_object("radiusWidget5").set_events(mask)
    
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
        
    # Opens the user manual in the default pdf application
    def showManual(self, *args):
        if sys.platform.startswith('darwin'):
            subprocess.call(('open', "manual/Manual.pdf"))
        elif os.name == 'nt': # For Windows
            os.startfile("manual\Manual.pdf")
        elif os.name == 'posix': # For Linux
            env = dict(os.environ)  # make a copy of the environment
            lp_key = 'LD_LIBRARY_PATH'  # for GNU/Linux and *BSD.
            lp_orig = env.get(lp_key + '_ORIG')
            if lp_orig is not None:
                env[lp_key] = lp_orig  # restore the original, unmodified value
            else:
                # This happens when LD_LIBRARY_PATH was not set.
                # Remove the env var as a last resort:
                env.pop(lp_key, None)
            # 'Clean' some other variables so they aren't using astrastack paths
            env.pop("GTK_PATH", None)
            env.pop("GTK_DATA_PREFIX", None)
            env.pop("XDG_DATA_DIRS", None)
            env.pop("GIO_MODULE_DIR", None)
            env.pop("PANGO_LIBDIR", None)
            env.pop("GDK_PIXBUF_MODULE_FILE", None)
            env.pop("GI_TYPELIB_PATH", None)
            # Now open the file
            subprocess.Popen(('xdg-open', "manual/Manual.pdf"), env=env)
        
    # Disable inputs
    def disableUI(self):
        self.builder.get_object("tabs").set_sensitive(False)
        self.builder.get_object("toolbar").set_sensitive(False)
        
    # Enable inputs
    def enableUI(self):
        self.builder.get_object("tabs").set_sensitive(True)
        self.builder.get_object("toolbar").set_sensitive(True)
        
    # The following is needed to forcibly refresh the value spacing of the slider
    def fixFrameSliderBug(self):
        self.frameScale.set_value_pos(Gtk.PositionType.RIGHT)
        self.frameScale.set_value_pos(Gtk.PositionType.LEFT)
        
    # Sets the number of threads to use
    def setThreads(self, *args):
        def initPool(method=None):
            if(method == "spawn"):
                GLib.idle_add(self.disableUI)
            g.nThreads = int(self.cpus.get_value())
            if(g.pool is not None):
                g.pool.shutdown()
            g.pool = ProcessPoolExecutor(g.nThreads)
            
            # This seems like the most reliable way to get the pid of pool processes
            self.pids = []
            before = list(map(lambda p : p.pid, active_children()))
            g.pool.submit(dummy, ()).result()
            after = list(map(lambda p : p.pid, active_children()))
            for pid in after:
                if(pid not in before):
                    self.pids.append(pid)
            if(method == "spawn"):
                GLib.idle_add(self.enableUI)
        
        # Behave a bit differently depending on platform
        if(get_start_method() == "spawn"):
            thread = Thread(target=initPool, args=(get_start_method(),))
            thread.start()
        else:
            initPool()
        
    # Checks github to see if there is a new version available
    def checkNewVersion(self):
        def callUrl():
            try:
                if(not g.TESTING):
                    contents = urllib.request.urlopen("https://api.github.com/repos/Finalfantasykid/AstraStack/releases/latest").read()
                    obj = json.loads(contents)
                    if(version.parse(obj['name']) > version.parse(UI.VERSION)):
                        self.newVersionUrl = "https://astrastack.ca/?page=download"
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
    
    # Shows preview image in file chooser dialog
    def updatePreview(self, dialog):
        path = dialog.get_preview_filename()
        pixbuf = None
        if(path != None and os.path.isfile(path)):
            try:
                # First try as image
                video = Video()
                img = cv2.cvtColor(video.getFrame(path, path, g.colorMode), cv2.COLOR_BGR2RGB).astype('uint8')
                height, width = img.shape[:2]
                
                z = img.tobytes()
                Z = GLib.Bytes.new(z)
                
                pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(Z, GdkPixbuf.Colorspace.RGB, False, 8, width, height, width*3)
            except Exception:
                try:
                    # Now try as video
                    video = Video()
                    img = cv2.cvtColor(video.getFrame(path, 0, g.colorMode), cv2.COLOR_BGR2RGB).astype('uint8')
                    height, width = img.shape[:2]
                    
                    z = img.tobytes()
                    Z = GLib.Bytes.new(z)
                    
                    pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(Z, GdkPixbuf.Colorspace.RGB, False, 8, width, height, width*3)
                except Exception:
                    pass
        if(pixbuf is not None):
            # Valid pixbuf
            maxwidth, maxheight = 250, 250
            width, height = pixbuf.get_width(), pixbuf.get_height()
            scale= min(maxwidth/width, maxheight/height)
            if(scale<1):
                width, height= int(width*scale), int(height*scale)
                pixbuf= pixbuf.scale_simple(width, height, GdkPixbuf.InterpType.BILINEAR)

            dialog.get_preview_widget().set_size_request(width + 10, height + 10)
            dialog.get_preview_widget().set_from_pixbuf(pixbuf)
            dialog.set_preview_widget_active(True)
        else:
            dialog.set_preview_widget_active(False)
    
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
                g.guessedColorMode = self.video.guessColorMode(g.file)
                img = cv2.imread(g.file)
                h, w = img.shape[:2]
                if(not self.checkMemory(w, h)):
                    raise MemoryError()
                
                self.window.set_title(path.split(g.file)[1] + " - " + UI.TITLE)
                self.saveDialog.set_current_name("")
                
                self.updateAutoColorModeText()
                
                self.sharpen = Sharpen(g.file, True)
                self.builder.get_object("alignTab").set_sensitive(False)
                self.builder.get_object("stackTab").set_sensitive(False)
                self.builder.get_object("processTab").set_sensitive(True)
                self.tabs.set_current_page(UI.SHARPEN_TAB)
                self.frame.set_from_file(g.file)
            except MemoryError as error:
                pass
            except: # Open Failed
                self.showErrorDialog("There was an error opening the image, make sure it is a valid image.")
            self.enableUI()
            
    # Opens the file chooser to save the final image
    def saveFileDialog(self, *args):
        self.saveDialog.set_current_folder(path.expanduser("~"))
        if(self.saveDialog.get_current_name() == ""):
            # Set default file to save if empty
            if(isinstance(g.file, list)):
                sList = g.file
                self.saveDialog.set_current_name(Path(sList[0]).stem + "_" + Path(sList[-1]).stem + ".png")
            else:
                self.saveDialog.set_current_name(Path(g.file).stem + ".png")
        def start():
            response = self.saveDialog.run()
            if(response == Gtk.ResponseType.OK):
                fileName = self.saveDialog.get_filename()
                try:
                    while(self.processThread != None and self.processThread.is_alive()):
                        time.sleep(0.1) # Make sure the file isn't still being processed
                    bitDepth = self.bitDepth.get_active_text()
                    if(bitDepth == "8-bit"):
                        cv2.imwrite(fileName, cv2.cvtColor(np.around(self.sharpen.finalImage).astype('uint8'), cv2.COLOR_RGB2BGR))
                    elif(bitDepth == "16-bit"):
                        if(fileName.endswith(".png") or fileName.endswith(".tif") or fileName.endswith(".tiff")):
                            cv2.imwrite(fileName, cv2.cvtColor(np.around(self.sharpen.finalImage*255).astype('uint16'), cv2.COLOR_RGB2BGR))
                        else:
                            self.showErrorDialog("Only .png and .tif are supported for 16-bit images")
                            start()
                except: # Save Failed
                    self.showErrorDialog("There was an error saving the image, make sure it is a valid file extension.")
        start()
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
            g.areaOfInterestP1 = (0, 0)
            g.areaOfInterestP2 = (0, 0)
            g.reference = self.video.sharpest
            self.frameSlider.set_value(self.video.sharpest)

            self.updateAutoColorModeText()
            self.setReference()
            self.setStartFrame()
            self.setEndFrame()
            self.setDriftPoint()
            self.enableUI()
            if(isinstance(g.file, list)):
                sList = g.file
                self.window.set_title(path.split(sList[0])[1] + " ... " + path.split(sList[-1])[1] +  " - " + UI.TITLE)
            else:
                self.window.set_title(path.split(g.file)[1] + " - " + UI.TITLE)
            self.saveDialog.set_current_name("")

            self.builder.get_object("alignTab").set_sensitive(True)
            self.builder.get_object("stackTab").set_sensitive(False)
            self.builder.get_object("processTab").set_sensitive(False)
            self.stack = None
        GLib.idle_add(update)
        
    # Opens the file chooser to open load a psf file (g.deconvolveCustomFile)
    def openPSF(self, *args):
        self.openDialog.set_current_folder(path.expanduser("~"))
        self.openDialog.set_select_multiple(False)
        self.openDialog.set_filter(self.builder.get_object("imageFilter"))
        response = self.openDialog.run()
        self.openDialog.hide()
        if(response == Gtk.ResponseType.OK):
            try:
                img = cv2.imread(self.openDialog.get_filename(), cv2.IMREAD_GRAYSCALE)
                if(max(img.shape) > 100):
                    raise Exception
                if(len(np.unique(img)) <= 1):
                    raise Exception
                img = np.float32(img)/255
                cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                g.deconvolveCustomFile = img
            except: # Open Failed
                self.showErrorDialog("There was an error opening the PSF, make sure it is a valid image.  The PSF should be no larger than 100x100.")
            self.sharpenImage(self.builder.get_object("deconvolveCustomFile"))
       
    # Updates the Auto color mode text     
    def updateAutoColorModeText(self):
        colorMode = self.colorMode.get_active()
        self.colorMode.remove(0)
        if(g.guessedColorMode == Video.COLOR_RGB):
            self.colorMode.prepend_text("Auto (RGB)")
        elif(g.guessedColorMode == Video.COLOR_GRAYSCALE):
            self.colorMode.prepend_text("Auto (Grayscale)")
        elif(g.guessedColorMode == Video.COLOR_RGGB):
            self.colorMode.prepend_text("Auto (RGGB)")
        elif(g.guessedColorMode == Video.COLOR_GRBG):
            self.colorMode.prepend_text("Auto (GRBG)")
        elif(g.guessedColorMode == Video.COLOR_GBRG):
            self.colorMode.prepend_text("Auto (GBRG)")
        elif(g.guessedColorMode == Video.COLOR_BGGR):
            self.colorMode.prepend_text("Auto (BGGR)")
        if(colorMode == Video.COLOR_AUTO):
            self.colorMode.set_active(0)
        
    # Called when the tab is changed.  Updates parts of the UI based on the tab
    def changeTab(self, notebook, page, page_num, user_data=None):
        self.colorMode.set_sensitive(True)
        if((page_num == UI.LOAD_TAB or page_num == UI.ALIGN_TAB) and self.video is not None):
            self.frameSlider.set_value(0)
            self.frameSlider.set_upper(max(0,len(self.video.frames)-1))
            self.setStartFrame()
            self.setEndFrame()
            if(len(self.video.frames) > 0):
                self.frameScale.show()
            self.updateImage(None, page_num)
        elif(page_num == UI.STACK_TAB and self.align is not None):
            self.frameSlider.set_lower(0)
            self.frameSlider.set_upper(max(0,len(self.align.tmats)-1))
            self.frameSlider.set_value(0)
            if(len(self.video.frames) > 0):
                self.frameScale.show()
            self.updateImage(None, page_num)
        elif(page_num == UI.SHARPEN_TAB and self.sharpen is not None):
            self.frameScale.hide()
            self.colorMode.set_sensitive(False)
            self.sharpenImage()
        self.fixFrameSliderBug()
    
    # Changes the image frame to the frameSlider position    
    def updateImage(self, adjustment=None, page_num=None):
        if(self.video is None):
            return
        if(page_num is None):
            page_num = self.tabs.get_current_page()
        if(page_num == UI.LOAD_TAB or page_num == UI.ALIGN_TAB):
            videoIndex = int(self.frameSlider.get_value())
            if(len(self.video.frames) == 0):
                # Single Image
                img = cv2.cvtColor(self.video.getFrame(g.file, g.file, (g.colorMode or g.guessedColorMode)), cv2.COLOR_BGR2RGB).astype(np.uint8)
            else:
                # Video/Sequence
                img = cv2.cvtColor(self.video.getFrame(g.file, self.video.frames[videoIndex], (g.colorMode or g.guessedColorMode)), cv2.COLOR_BGR2RGB).astype(np.uint8)
            height, width = img.shape[:2]
            
            z = img.tobytes()
            Z = GLib.Bytes.new(z)
            
            pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(Z, GdkPixbuf.Colorspace.RGB, False, 8, width, height, width*3)
            self.frame.set_from_pixbuf(pixbuf)
        elif(page_num == UI.STACK_TAB and self.stack is not None):
            tmat = self.stack.tmats[int(self.frameSlider.get_value())]
            videoIndex = tmat[0]
            M = tmat[1]
            img = self.video.getFrame(g.file, videoIndex, (g.colorMode or g.guessedColorMode)).astype(np.uint8)
            if(g.autoCrop):
                ref = self.stack.refBG.astype(np.uint8)
            else:
                ref = None
            img = transform(img, ref, M,
                            self.align.minX, self.align.maxX, self.align.minY, self.align.maxY, 
                            g.drizzleFactor, g.drizzleInterpolation)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            
            z = img.tobytes()
            Z = GLib.Bytes.new(z)
            
            pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(Z, GdkPixbuf.Colorspace.RGB, False, 8, width, height, width*3)
            self.frame.set_from_pixbuf(pixbuf)
            
    # Draws a rectangle where the area of interest is
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
            
            # Area of Interest
            px1 = min(g.areaOfInterestP1[0], g.areaOfInterestP2[0])
            py1 = min(g.areaOfInterestP1[1], g.areaOfInterestP2[1])
            
            px2 = max(g.areaOfInterestP1[0], g.areaOfInterestP2[0])
            py2 = max(g.areaOfInterestP1[1], g.areaOfInterestP2[1])
            
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
                # Draw Area of Interest Rectangle
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
        
    # Updates the PSF Image to the current configuration for deblurring        
    def updatePSFImage(self, *args):
        sz = 110
        psf = np.zeros((sz,sz), np.float32)
        count = 0
        if(g.deconvolveCircular and g.deconvolveCircularDiameter > 1):
            # Circular
            psf += defocus_kernel(int(g.deconvolveCircularDiameter), sz)
            count += 1
        if(g.deconvolveGaussian and g.deconvolveGaussianDiameter > 1):
            # Gaussian
            kern = gaussian_kernel(int(g.deconvolveGaussianDiameter), int(g.deconvolveGaussianSpread), sz)
            psf += cv2.normalize(kern, kern, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            count += 1
        if(g.deconvolveLinear and g.deconvolveLinearDiameter > 1):
            # Stretch&Rotate the Gaussian/Circular
            psf = stretchPSF(psf, (g.deconvolveLinearDiameter/max(g.deconvolveGaussianDiameter,g.deconvolveCircularDiameter)), g.deconvolveLinearAngle)
        if(count == 0 and g.deconvolveLinear and g.deconvolveLinearDiameter > 1):
            # Linear
            psf += motion_kernel(int(g.deconvolveLinearAngle), int(g.deconvolveLinearDiameter), sz)
            count += 1
        if(g.deconvolveCustom and g.deconvolveCustomFile is not None):
            # Custom
            kern = g.deconvolveCustomFile
            h, w = kern.shape[:2]
            kern = cv2.copyMakeBorder(kern, math.ceil((sz-h)/2), math.floor((sz-h)/2), math.ceil((sz-w)/2), math.floor((sz-w)/2), cv2.BORDER_CONSTANT)
            psf += kern
            count += 1
            
        if(count > 0):
            psf /= count
        else:
            psf[int(sz/2)][int(sz/2)] = 1
        
        psf = cv2.normalize(psf, psf, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)    
        psf = np.uint8(psf*255)
        
        psf = cv2.cvtColor(psf, cv2.COLOR_GRAY2RGB)
        
        z = psf.tobytes()
        Z = GLib.Bytes.new(z)
        
        pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(Z, GdkPixbuf.Colorspace.RGB, False, 8, sz, sz, sz*3)
        self.psfImage.set_from_pixbuf(pixbuf)
        
    # Sets the reference frame to the current visible frame
    def setReference(self, *args):
        g.reference = str(int(self.frameSlider.get_value()))
        self.builder.get_object("referenceLabel").set_text(g.reference)
        self.builder.get_object("alignButton").set_sensitive(True)
        
    # Updates the progress bar
    def setProgress(self, i=0, total=0, text=""):
        def update():
            if(total == 0):
                self.progressBox.hide()
            else:
                self.progressBox.show()
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
        self.clickedAreaOfInterest = False
        self.setDriftPoint()
        self.frameSlider.set_value(g.startFrame)
        self.clickedDriftP1 = True
        self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.CROSSHAIR))
        
    # Drift Point 2 Button Clicked
    def clickDriftP2(self, *args):
        self.clickedDriftP1 = False
        self.clickedDriftP2 = False
        self.clickedAreaOfInterest = False
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
            self.clickedAreaOfInterest = False
            self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.LEFT_PTR))
            self.overlay.queue_draw()
            
    # Reset Drift Point 2 to (0, 0)
    def resetDriftP2(self, widget, event):
        if(event.button == 3): # Right Click
            g.driftP2 = (0, 0)
            self.clickedDriftP1 = False
            self.clickedDriftP2 = False
            self.clickedAreaOfInterest = False
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
        
    # Area of Interest button clicked
    def clickAreaOfInterest(self, *args):
        g.areaOfInterestP1 = (0, 0)
        g.areaOfInterestP2 = (0, 0)
        self.clickedDriftP1 = False
        self.clickedDriftP2 = False
        self.clickedAreaOfInterest = True
        self.frameSlider.set_value(g.startFrame)
        self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.CROSSHAIR))
        self.overlay.queue_draw()
        
    # Reset Area of Interest to (0, 0)
    def resetAreaOfInterest(self, widget, event):
        if(event.button == 3): # Right Click
            g.areaOfInterestP1 = (0, 0)
            g.areaOfInterestP2 = (0, 0)
            self.clickedDriftP1 = False
            self.clickedDriftP2 = False
            self.clickedAreaOfInterest = False
            self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.LEFT_PTR))
            self.overlay.queue_draw()
        
    # First point int the Area of Interest clicked, drag started
    def dragBegin(self, *args):
        if(self.clickedAreaOfInterest):
            g.areaOfInterestP1 = self.mousePosition
        
    # Mouse released after dragging Area of Interest
    def dragEnd(self, *args):
        if(self.clickedAreaOfInterest):
            g.areaOfInterestP2 = self.mousePosition
            self.clickedAreaOfInterest = False
            self.window.get_window().set_cursor(Gdk.Cursor(Gdk.CursorType.LEFT_PTR))
    
    # Called when the mouse moves over the frame
    def updateMousePosition(self, *args):
        pointer = self.frame.get_pointer()
        self.mousePosition = (min(max(0, pointer.x), self.frame.get_allocation().width), min(max(0, pointer.y), self.frame.get_allocation().height))
        if(self.clickedAreaOfInterest):
            if(g.areaOfInterestP1 != (0, 0)):
                g.areaOfInterestP2 = self.mousePosition
            self.overlay.queue_draw()
    
    # Sets whether or not to normalize the frames during alignment
    def setNormalize(self, *args):
        g.normalize = self.normalize.get_active()
        
    # Sets whether or not to align channels separately
    def setAlignChannels(self, *args):
        g.alignChannels = self.alignChannels.get_active()
    
    # Sets the color mode of the input
    def setColorMode(self, *args):
        if(self.colorMode.is_sensitive()):
            g.colorMode = self.colorMode.get_active()
            self.updateImage()
    
    # Sets the type of transformation
    def setTransformation(self, *args):
        text = self.transformation.get_active_text()
        if(text == "None"):
            g.transformation = -1
        elif(text == "Translation"):
            g.transformation = StackReg.TRANSLATION
        elif(text == "Rigid Body"):
            g.transformation = StackReg.RIGID_BODY
        elif(text == "Scaled Rotation"):
            g.transformation = StackReg.SCALED_ROTATION
        elif(text == "Affine"):
            g.transformation = StackReg.AFFINE
            
    # Sets the drizzle scaling factor
    def setDrizzleFactor(self, *args):
        text = self.drizzleFactor.get_active_text()
        if(text == "0.25X"):
            g.drizzleFactor = 0.25
        elif(text == "0.50X"):
            g.drizzleFactor = 0.50
        elif(text == "0.75X"):
            g.drizzleFactor = 0.75
        elif(text == "1.0X"):
            g.drizzleFactor = 1.0
        elif(text == "1.5X"):
            g.drizzleFactor = 1.5
        elif(text == "2.0X"):
            g.drizzleFactor = 2.0
        elif(text == "2.5X"):
            g.drizzleFactor = 2.5
        elif(text == "3.0X"):
            g.drizzleFactor = 3.0
        if(self.stack is not None):
            self.stack.generateRefBG()
        self.updateImage()
            
    # Sets the drizzle scaling factor
    def setDrizzleInterpolation(self, *args):
        text = self.drizzleInterpolation.get_active_text()
        if(text == "Nearest Neighbor"):
            g.drizzleInterpolation = cv2.INTER_NEAREST
        elif(text == "Bilinear"):
            g.drizzleInterpolation = cv2.INTER_LINEAR
        elif(text == "Bicubic"):
            g.drizzleInterpolation = cv2.INTER_CUBIC
        elif(text == "Lanczos"):
            g.drizzleInterpolation = cv2.INTER_LANCZOS4
        if(self.stack is not None):
            self.stack.generateRefBG()
        self.updateImage()
        
    # Sets whether or not to auto crop
    def setAutoCrop(self, *args):
        g.autoCrop = not self.autoCrop.get_active()
        self.updateImage()
            
    # Runs the Alignment step
    def clickAlign(self, *args):
        self.align = Align(self.video.frames[g.startFrame:g.endFrame+1])
        thread = Thread(target=self.align.run, args=())
        thread.start()
        self.disableUI()
        
    # Kills all pool processes
    def killPool(self):
        for pid in self.pids:
            if(psutil.pid_exists(pid)):
                p = psutil.Process(pid)
                p.kill()
        
    # Stops the current action being performed
    def stopProcessing(self, *args):
        self.killPool()
        g.pool = None
        self.setThreads()
        self.setProgress()
        self.enableUI()
        
    # Called when the Alignment is complete
    def finishedAlign(self):
        def update():
            self.stack = Stack(self.align.tmats)
            self.tabs.next_page()
            self.enableUI()
            self.builder.get_object("alignTab").set_sensitive(True)
            self.builder.get_object("stackTab").set_sensitive(True)
            self.builder.get_object("processTab").set_sensitive(False)
            self.limit.set_upper(len(self.align.tmats))
            self.limit.set_value(int(len(self.align.tmats)/2))
            self.limitPercent.set_value(round(self.limit.get_value()/len(self.align.tmats)*100))
            self.setLimit()
            self.setLimitPercent()
        GLib.idle_add(update)
        
    # Sets the number of frames to use in the Stack
    def setLimit(self, *args):
        self.limitPercent.disconnect(self.limitPercentSignal)
        self.limit.set_upper(len(self.align.tmats))
        g.limit = int(self.limit.get_value())
        self.limitPercent.set_value(round(g.limit/len(self.align.tmats)*100))
        self.limitPercentSignal = self.limitPercent.connect("value-changed", self.setLimitPercent)
        
    # Sets the number of frames to use in the Stack
    def setLimitPercent(self, *args):
        limitPercent = self.limitPercent.get_value()/100
        self.limit.set_value(round(limitPercent*len(self.align.tmats)))
            
    # Stack Button clicked
    def clickStack(self, *args):
        try:
            self.stack.checkMemory()
            thread = Thread(target=self.stack.run, args=())
            thread.start()
            self.disableUI()
        except MemoryError as error:
            self.enableUI()
        
    # Called when the stack is complete
    def finishedStack(self):
        def update():
            self.sharpen = Sharpen(self.stack.stackedImage)
            self.tabs.next_page()
            self.enableUI()
            self.builder.get_object("alignTab").set_sensitive(True)
            self.builder.get_object("stackTab").set_sensitive(True)
            self.builder.get_object("processTab").set_sensitive(True)
        GLib.idle_add(update)
        
    # Sharpens the final Stacked image
    def sharpenImage(self, *args):
        if(g.file is None):
            return
        self.processSpinner.start()
        self.processSpinner.show()
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
        
        g.deconvolveCircular = self.builder.get_object("deconvolveCircular").get_active()
        g.deconvolveGaussian = self.builder.get_object("deconvolveGaussian").get_active()
        g.deconvolveLinear = self.builder.get_object("deconvolveLinear").get_active()
        g.deconvolveCustom = self.builder.get_object("deconvolveCustom").get_active()
        
        g.deconvolveCircularDiameter = self.builder.get_object("deconvolveCircularDiameter").get_value()
        g.deconvolveCircularAmount = self.builder.get_object("deconvolveCircularAmount").get_value()
        g.deconvolveGaussianDiameter = self.builder.get_object("deconvolveGaussianDiameter").get_value()
        g.deconvolveGaussianAmount = self.builder.get_object("deconvolveGaussianAmount").get_value()
        g.deconvolveGaussianSpread = self.builder.get_object("deconvolveGaussianSpread").get_value()
        g.deconvolveLinearDiameter = self.builder.get_object("deconvolveLinearDiameter").get_value()
        g.deconvolveLinearAmount = self.builder.get_object("deconvolveLinearAmount").get_value()
        g.deconvolveLinearAngle = self.builder.get_object("deconvolveLinearAngle").get_value()
        g.deconvolveCustomAmount = self.builder.get_object("deconvolveCustomAmount").get_value()
        
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
            processDeblur = self.sharpen.processDeblurAgain
            processColor = True
        elif(len(args) > 0 and (self.builder.get_object("deconvolveCircular") == args[0] or
                                self.builder.get_object("deconvolveGaussian") == args[0] or
                                self.builder.get_object("deconvolveLinear") == args[0] or
                                self.builder.get_object("deconvolveCustom") == args[0] or
                                self.builder.get_object("deconvolveCircularDiameter") == args[0] or
                                self.builder.get_object("deconvolveCircularAmount") == args[0] or
                                self.builder.get_object("deconvolveGaussianDiameter") == args[0] or
                                self.builder.get_object("deconvolveGaussianAmount") == args[0] or
                                self.builder.get_object("deconvolveGaussianSpread") == args[0] or
                                self.builder.get_object("deconvolveLinearDiameter") == args[0] or
                                self.builder.get_object("deconvolveLinearAmount") == args[0] or
                                self.builder.get_object("deconvolveLinearAngle") == args[0] or
                                self.builder.get_object("deconvolveCustomFile") == args[0] or
                                self.builder.get_object("deconvolveCustomAmount") == args[0])):
            processAgain = self.sharpen.processAgain
            processDeblur = True
            processColor = False
        else:
            processAgain = True
            processDeblur = False
            processColor = False
        
        if(self.sharpen is None):
            if(self.stack is not None):
                self.sharpen = Sharpen(self.stack.stackedImage)
            else:
                self.sharpen = Sharpen(g.file, True)
        if(self.processThread != None and self.processThread.is_alive()):
            self.sharpen.processAgain = processAgain
            self.sharpen.processDeblurAgain = processDeblur
            self.sharpen.processColorAgain = processColor
        else:
            self.processThread = Thread(target=self.sharpen.run, args=(processAgain, processDeblur, processColor))
            self.processThread.start()
        self.updatePSFImage(args)
    
    # Called when sharpening is complete
    def finishedSharpening(self):
        def update():
            z = np.around(self.sharpen.finalImage).astype('uint8').tobytes()
            Z = GLib.Bytes.new(z)
            pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(Z, GdkPixbuf.Colorspace.RGB, False, 8, self.sharpen.w, self.sharpen.h, self.sharpen.w*3)
            self.frame.set_from_pixbuf(pixbuf)
            self.processSpinner.stop()
            self.processSpinner.hide()
        GLib.idle_add(update)

    # Closes the application
    def close(self, *args):
        self.killPool()
        Gtk.main_quit()

# Used to initialize pids
def dummy(*args):
    return True

def run():
    # Newer versions of Adwaita scalable icons don't work well with older librsvg.
    # This can be removed when no longer being built with an older librsvg
    Gtk.IconTheme.get_default().prepend_search_path('share_override/icons')

    g.ui = UI()

    Gtk.main()
