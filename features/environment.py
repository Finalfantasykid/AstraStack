import os
import sys
import time
import shutil
from threading import Thread
from gi import require_version
require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
sys.path.insert(0, os.getcwd())

from UI import UI
from Globals import g

def before_all(context):
    # First delete files if they exist
    try:
        shutil.rmtree("features/testFiles/tmp/")
    except:
        pass
    # Then create it again
    try:
        os.mkdir("features/testFiles/tmp/")
    except:
        pass
    
def before_scenario(context, scenario):
    g.ui = UI()
    g.ui.builder.get_object("wavelets").set_expanded(True)
    g.ui.builder.get_object("deblur").set_expanded(True)
    g.ui.builder.get_object("colors").set_expanded(True)
    def run():
        Gtk.main()
    thread = Thread(target=run, args=())
    thread.start()
    time.sleep(0.5)
    
def after_scenario(context, scenario):
    g.ui.window.close()
    time.sleep(0.1)
    
def after_all(context):
    try:
        pass
        shutil.rmtree("features/testFiles/tmp/")
    except:
        pass
    
