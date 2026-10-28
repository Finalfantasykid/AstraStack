# AstraStack
Astronomy Stacking application - [Manual](https://github.com/Finalfantasykid/AstraStack/blob/master/manual/Manual.pdf) - [Website](https://astrastack.ca/)

AstraStack is an easy to use application used to stack videos or images of planets, the sun, moon, deep sky objects and more. Stacking is a technique used to improve the quality of the image by aligning each frame and then averaging the best frames to create a low noise version of the image which can then be sharpened.

[![Get it from the Snap Store](https://snapcraft.io/static/images/badges/en/snap-store-black.svg)](https://snapcraft.io/astrastack)

# Development Installation
To install you should first set up a virtual environment

```
virtualenv -p python3 env
```

Then activate the virtual environment

```
source env/bin/activate
```

Install required python packages with pip.  You may need to first install numpy before running the requirements install if you don't have the numpy system package installed.


```
pip install numpy
pip install -r requirements.txt
```

You may need to install some additional system packages, ie: libcairo2-dev, pkg-config, python3-dev, libgirepository-2.0-dev

You can run the application with

```
python AstraStack.py
```

# Developer Instructions
There are other packages which are used for developers

```
pip install -r requirements_dev.txt
```

## Build Instructions
Building primarily uses pyinstaller to package into an executable.

```
./build.sh
```

The project can also be built in Windows using a mingw64 installation.  You can install the python packages via the pacman package manager that it comes with, as well as with pip.  Inno Setup is also used to create an installer file.

## Unit Testing

Unit tests use `pytest` and live under `tests/`.  Run them using `pytest`:

```
pytest
```

## End-To-End Testing

E2E testing leverages the `behave` framework to simulate Gtk actions in the UI.
Test scenarios are defined under `features/`, and you can run the full test
suite by simply running `behave`:

```
behave
```

Alternatively run it headless:

```
# Wayland:
wlheadless-run -- behave 2>/dev/null
# X11:
xvfb-run behave
```

Debugging a specific scenario is best accomplished by giving a persisted output
path and passing the line number to the problematic scenario:

```
ASTRASTACK_TEST_OUTPUTS=/tmp/astrastack-outputs behave features/single.feature:3
```

Running the full E2E test suite (e.g. in CI) is normally best accomplished
using parallel headless mode and is about 2-3x faster than bare `behave`:

```
# Prerequisite: `weston` OS package.
scripts/run_behave.sh
```

# References
1. RegiStax - https://www.astronomie.be/registax/
1. PyStackReg - https://pypi.org/project/pystackreg/
1. TurboReg - http://bigwww.epfl.ch/thevenaz/turboreg/
1. StackReg - http://bigwww.epfl.ch/thevenaz/StackReg/
1. A Pyramid Approach to Subpixel Registration Based on Intensity - http://bigwww.epfl.ch/publications/thevenaz9801.html
1. Image Sharpening using Unsharp Masking and Wavelet Transform - http://ijarcsms.com/docs/paper/volume2/issue6/V2I6-0003.pdf
