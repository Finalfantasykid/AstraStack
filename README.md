# AstraStack
Astronomy Stacking application

# Development Installation
To install you should first set up a virtual environment

```
$ virtualenv -p python3 env
```

Then activate the virtual environment

```
$ source env/bin/activate
```

Install required python packages with pip


```
$ pip install -r requirements.txt
```

You may need to install some additional system packages, like some python development packages.

You can run the application with

```
$ python UI.py
```

# References
1. PyStackReg - https://pypi.org/project/pystackreg/
    1. TurboReg - http://bigwww.epfl.ch/thevenaz/turboreg/
    1. StackReg - http://bigwww.epfl.ch/thevenaz/StackReg/
    1. A Pyramid Approach to Subpixel Registration Based on Intensity - http://bigwww.epfl.ch/publications/thevenaz9801.html
1. Image Sharpening using Unsharp Masking and Wavelet Transform - http://ijarcsms.com/docs/paper/volume2/issue6/V2I6-0003.pdf
