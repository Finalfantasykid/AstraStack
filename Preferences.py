import json
import os
import pyglet

class Preferences:
    
    FILE = "preferences.json"
    
    def __init__(self):
        from UI import UI
        self.json = {}
        self.dir = pyglet.resource.get_settings_path(UI.TITLE) + "/"
        self.file = self.dir + Preferences.FILE
        try:
            if not os.path.exists(self.dir):
                os.makedirs(self.dir)
            if(os.path.isfile(self.file)): 
                with open(self.file) as f:
                    self.json = json.loads(f.read())
        except:
            print("Error loading preferences")
        
    def set(self, field, value):
        try:
            self.json[field] = value
            with open(self.file, "w") as f:
                f.write(json.dumps(self.json))
        except:
            print("Error saving preferences");
    
    def get(self, field, default=""):
        if(field in self.json):
            return self.json[field]
        return default
