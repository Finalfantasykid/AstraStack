import os
import sys
from behave import *
import time
import cv2
import numpy as np
from threading import Thread
from gi import require_version
require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
sys.path.insert(0, os.getcwd())

from UI import UI
from Globals import g

tabMap = {"Load": UI.LOAD_TAB,
          "Align": UI.ALIGN_TAB,
          "Stack": UI.STACK_TAB,
          "Process": UI.SHARPEN_TAB}

def delay(delay=0.1):
    time.sleep(delay)
    
@given(u'I wait "{ms}"')
def wait(context, ms):
    delay(int(ms)/1000)
    
@given(u'I wait until active tab is "{tab}"')
def waitUntil(context, tab):
    while(g.ui.builder.get_object("tabs").get_current_page() != tabMap[tab]):
        delay()

@given(u'I am on tab "{tab}"')
def changeTab(context, tab):
    g.ui.builder.get_object("tabs").set_current_page(tabMap[tab])
    delay()

@given(u'I press "{button}"')
def pressButton(context, button):
    def update():
        g.ui.builder.get_object(button).clicked()
    GLib.idle_add(update)
    delay()

@given(u'I async press "{button}"')
def asyncPressButton(context, button):
    def run():
        def update():
            g.ui.builder.get_object(button).clicked()
        GLib.idle_add(update)
    thread = Thread(target=run, args=())
    thread.start()
    delay()
    
@given(u'I choose file "{file}" from "{dialog}"')
def chooseFile(context, file, dialog):
    window = g.ui.builder.get_object(dialog)
    while(not window.is_visible()):
        delay()
    def update1():
        window.select_filename(os.getcwd() + "/features/testFiles/" + file)
    def update2():
        if("*" in file):
            window.select_all()
        if(window.get_action() == Gtk.FileChooserAction.SAVE):
            window.set_current_name(file.split("/")[-1])
    def update3():
        window.get_widget_for_response(Gtk.ResponseType.OK).clicked()
    GLib.idle_add(update1)
    delay()
    GLib.idle_add(update2)
    delay()
    GLib.idle_add(update3)
    while(window.is_visible()):
        delay()
    
@given(u'I set "{adjustment}" to "{value}"')
def setAdjustment(context, adjustment, value):
    def update():
        g.ui.builder.get_object(adjustment).set_value(float(value))
    GLib.idle_add(update)
    delay()
    
@given(u'I select "{value}" from "{combo}"')
def setCombo(context, value, combo):
    model = g.ui.builder.get_object(combo).get_model()
    iter = model.get_iter_first()
    id = None
    i = 0
    while True:
        val = model.get_value(iter, 0)
        if(val == value):
            id = i
        iter = model.iter_next(iter)
        i += 1
        if iter is None:
            break
    assert id is not None, "Item '" + value + "' not found"
    def update():
        g.ui.builder.get_object(combo).set_active(id)
    GLib.idle_add(update)
    delay()
    
@given(u'I set point "{var}" to "{point}"')
def setPoint(context, var, point):
    point = point.split(",")
    point = [int(x) for x in point]
    def update1():
        if(var == "driftP1"):
            g.ui.clickDriftP1()
        elif(var == "driftP2"):
            g.ui.clickDriftP2()
        elif(var == "areaOfInterestP1"):
            g.ui.clickAreaOfInterest()
        elif(var == "areaOfInterestP2"):
            pass
        else:
            assert False, var + " is not a valid variable"
    def update2():
        g.ui.mousePosition = point
        if(var == "driftP1" or var == "driftP2"):
            g.ui.setDriftPoint()
        elif(var == "areaOfInterestP1"):
            g.ui.dragBegin()
        elif(var == "areaOfInterestP2"):
            g.ui.dragEnd()
        g.ui.updateImage()
    GLib.idle_add(update1)
    delay()
    GLib.idle_add(update2)
    delay()
    
@then(u'"{file1}" and "{file2}" should be equal')
def compareImages(context, file1, file2):
    img1 = cv2.imread("features/testFiles/" + file1, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread("features/testFiles/" + file2, cv2.IMREAD_UNCHANGED)
    assert np.array_equal(img1, img2), "'" + file1 + "' and '" + file2 + "' are not equal"
