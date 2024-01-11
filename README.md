# AstraStack
Astronomy Stacking application - [Manual](https://github.com/Finalfantasykid/AstraStack/blob/master/manual/Manual.pdf) - [Website](https://astrastack.ca/)

AstraStack is an easy to use application used to stack videos or images of planets, the sun, moon, deep sky objects and more. Stacking is a technique used to improve the quality of the image by aligning each frame and then averaging the best frames to create a low noise version of the image which can then be sharpened.

[![Get it from the Snap Store](https://snapcraft.io/static/images/badges/en/snap-store-black.svg)](https://snapcraft.io/astrastack)

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

# Developer Instructions
There are other packages which are used for developers

```
$ pip install -r requirements_dev.txt
```

## Build Instructions
Building primarily uses pyinstaller to package into an executable.

```
$ ./build.sh
```

The project can also be built in Windows using a mingw64 installation.  You can install the python packages via the pacman package manager that it comes with, as well as with pip.  Inno Setup is also used to create an installer file.

## Testing
Testing uses the behave package and Gtk instructions are 'simulated' during testing.  To run the tests run the following:

```
$ behave
```

# References
1. RegiStax - https://www.astronomie.be/registax/
1. PyStackReg - https://pypi.org/project/pystackreg/
1. TurboReg - http://bigwww.epfl.ch/thevenaz/turboreg/
1. StackReg - http://bigwww.epfl.ch/thevenaz/StackReg/
1. A Pyramid Approach to Subpixel Registration Based on Intensity - http://bigwww.epfl.ch/publications/thevenaz9801.html
1. Image Sharpening using Unsharp Masking and Wavelet Transform - http://ijarcsms.com/docs/paper/volume2/issue6/V2I6-0003.pdf
