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

Install required python packages with pip.  You may need to first install numpy before running the requirements install if you don't have the numpy system package installed.


```
$ pip install numpy
$ pip install -r requirements.txt
```

You may need to install some additional system packages, ie: libcairo2-dev, pkg-config, python3-dev, libgirepository1.0-dev

You can run the application with

```
$ python AstraStack.py
```

# Build Instructions
In order to build the project for Linux, you will need to install pyinstaller using pip, and then run the build command

```
$ pip install pyinstaller
$ ./build.sh
```

The project can also be built in Windows using a mingw64 installation.  You can install the python packages via the pacman package manager that it comes with, as well as with pip.

# References
1. RegiStax - https://www.astronomie.be/registax/
1. PyStackReg - https://pypi.org/project/pystackreg/
1. TurboReg - http://bigwww.epfl.ch/thevenaz/turboreg/
1. StackReg - http://bigwww.epfl.ch/thevenaz/StackReg/
1. A Pyramid Approach to Subpixel Registration Based on Intensity - http://bigwww.epfl.ch/publications/thevenaz9801.html
1. Image Sharpening using Unsharp Masking and Wavelet Transform - http://ijarcsms.com/docs/paper/volume2/issue6/V2I6-0003.pdf
1. image_similarity - https://github.com/petermat/image_similarity
