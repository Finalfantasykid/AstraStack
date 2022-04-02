import json
import os
import pyglet
from Globals import *

class Preferences:
    
    FILE = "preferences.json"
    
    def __init__(self):
        from UI import UI
        self.json = {}
        self.dir = pyglet.resource.get_settings_path(UI.TITLE) + "/"
        self.file = self.dir + Preferences.FILE
        if(not g.TESTING):
            try:
                if not os.path.exists(self.dir):
                    os.makedirs(self.dir)
                if(os.path.isfile(self.file)): 
                    with open(self.file) as f:
                        self.json = json.loads(f.read())
            except:
                print("Error loading preferences")
        
    # Sets and saves the preference value
    def set(self, field, value):
        try:
            self.json[field] = value
            if(not g.TESTING):
                with open(self.file, "w") as f:
                    f.write(json.dumps(self.json))
        except:
            print("Error saving preferences");
    
    # Returns the value of the specified field.
    def get(self, field, default=""):
        if(field in self.json):
            return self.json[field]
        return default
