import tempfile

g = type('', (), {})()

g.tmp = tempfile.gettempdir() + "/AstraStack/"
