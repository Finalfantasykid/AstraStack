import os
import sys
from behave import *
import time
import cv2
import numpy as np
from threading import Thread
from multiprocessing import Value
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
    
def idle_add_delay(fn):
    updateDone = Value('b', False)
    def update(updateDone):
        fn()
        updateDone.value = True
    GLib.idle_add(update, updateDone)
    while(not updateDone.value):
        delay(0.01)
    delay(0.01) # Always delay just a bit
    
def widgetValidation(id):
    widget = g.ui.builder.get_object(id)
    i = 0
    while ((widget is None or
            not widget.is_visible() or
            not widget.is_sensitive()) and 
           i < 10):
        # Give it a chance to become valid before asserting
        delay(0.05)
        i += 1
    assert widget != None, f"The widget '{id}' does not exist"
    assert widget.is_visible(), f"The widget '{id}' is not visible"
    assert widget.is_sensitive(), f"The widget '{id}' is not sensitive"
    
@given(u'I wait "{ms}"')
def wait(context, ms):
    delay(int(ms)/1000)
    
@given(u'I wait until active tab is "{tab}"')
def waitUntil(context, tab):
    while(g.ui.builder.get_object("tabs").get_current_page() != tabMap[tab]):
        delay(0.01)

@given(u'I am on tab "{tab}"')
def changeTab(context, tab):
    widgetValidation("tabs")
    def update():
        g.ui.builder.get_object("tabs").set_current_page(tabMap[tab])
    idle_add_delay(update)

@given(u'I press "{button}"')
def pressButton(context, button):
    widgetValidation(button)
    def update():
        g.ui.builder.get_object(button).clicked()
    idle_add_delay(update)

@given(u'I async press "{button}"')
def asyncPressButton(context, button):
    widgetValidation(button)
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
        delay(0.01)
    delay() # Sometimes it crashes if going to the next step too quickly, so delay a bit
    def update1():
        window.select_filename(os.getcwd() + "/features/testFiles/" + file)
    def update2():
        if("*" in file):
            window.select_all()
        if(window.get_action() == Gtk.FileChooserAction.SAVE):
            window.set_current_name(file.split("/")[-1])
    def update3():
        window.get_widget_for_response(Gtk.ResponseType.OK).clicked()
    idle_add_delay(update1)
    idle_add_delay(update2)
    idle_add_delay(update3)
    while(window.is_visible()):
        delay(0.01)
    
@given(u'I set "{adjustment}" to "{value}"')
def setAdjustment(context, adjustment, value):
    def update():
        g.ui.builder.get_object(adjustment).set_value(float(value))
    idle_add_delay(update)
    
@given(u'I check "{checkbox}"')
def setCheck(context, checkbox):
    widgetValidation(checkbox)
    def update():
        g.ui.builder.get_object(checkbox).set_active(True)
    idle_add_delay(update)
    
@given(u'I uncheck "{checkbox}"')
def setUncheck(context, checkbox):
    widgetValidation(checkbox)
    def update():
        g.ui.builder.get_object(checkbox).set_active(False)
    idle_add_delay(update)
    
@given(u'I select "{value}" from "{combo}"')
def setCombo(context, value, combo):
    widgetValidation(combo)
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
    assert id is not None, f"Item '{value}' not found"
    def update():
        g.ui.builder.get_object(combo).set_active(id)
    idle_add_delay(update)
    
@given(u'I set point "{var}" to "{point}"')
def setPoint(context, var, point):
    point = point.split(",")
    point = [int(x) for x in point]
    assert (var == "driftP1" or var == "driftP2" or
            var == "areaOfInterestP1" or var == "areaOfInterestP2"), f"'{var}' is not a valid variable"
    def update1():
        if(var == "driftP1"):
            g.ui.clickDriftP1()
        elif(var == "driftP2"):
            g.ui.clickDriftP2()
        elif(var == "areaOfInterestP1"):
            g.ui.clickAreaOfInterest()
    def update2():
        g.ui.mousePosition = point
        if(var == "driftP1" or var == "driftP2"):
            g.ui.setDriftPoint()
        elif(var == "areaOfInterestP1"):
            g.ui.dragBegin()
        elif(var == "areaOfInterestP2"):
            g.ui.dragEnd()
        g.ui.updateImage()
    idle_add_delay(update1)
    idle_add_delay(update2)
    
@then(u'"{var}" should equal "{val}"')
def shouldEqual(context, var, val):
    if(hasattr(g, var)):
        assert str(getattr(g, var)) == str(val), f"'{var}' does not equal '{val}'"
    elif(isinstance(g.ui.builder.get_object(var), Gtk.Adjustment)):
        assert str(g.ui.builder.get_object(var).get_value()) == str(val), f"'{var}' does not equal '{val}'"
    elif(isinstance(g.ui.builder.get_object(var), Gtk.Label)):
        widgetValidation(var)
        assert str(g.ui.builder.get_object(var).get_text()) == str(val), f"'{var}' does not equal '{val}'"
    else:
        assert False, f"'{var}' does not exist"
    
@then(u'"{file1}" and "{file2}" should be equal')
def compareImages(context, file1, file2):
    img1 = cv2.imread("features/testFiles/" + file1, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread("features/testFiles/" + file2, cv2.IMREAD_UNCHANGED)
    assert np.array_equal(img1, img2), f"'{file1}' and '{file2}' are not equal"
