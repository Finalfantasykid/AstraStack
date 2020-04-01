#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    # Get location of libxcb.so.1 so it can be included
    LIBXCB=`ldconfig -p | grep libxcb.so.1 | grep x86-64 | sed 's/^.*=> //'`

    rm -fr dist
    pyinstaller --add-binary "$LIBXCB:." \
                AstraStack.py
    rm -fr dist/AstraStack/share/icons/
    rm -fr dist/AstraStack/share/themes/
    cp -r share/themes dist/AstraStack/share/
    cp -r share/icons dist/AstraStack/share/
    cp -r ui dist/
    cp scripts/astra-stack dist/
    cp scripts/astra-stack.desktop dist/
    cd dist
    tar cfz ../AstraStack.tar.gz *
elif [[ "$OSTYPE" == "msys" ]]; then
    rm -fr dist
    pyinstaller --windowed \
                --icon "ui/logo.ico" \
                --hidden-import "packaging.requirements" \
                --hidden-import "pkg_resources.py2_warn" \
                --add-binary "/mingw64/bin/opencv_videoio_ffmpeg420_64.dll:." \
                --exclude-module "FixTk" \
                --exclude-module "tcl" \
                --exclude-module "tk" \
                --exclude-module "_tkinter" \
                --exclude-module "tkinter" \
                --exclude-module "Tkinter" \
                AstraStack.py
    rm -fr dist/AstraStack/share/icons/
    rm -fr dist/AstraStack/share/themes/
    rm -fr dist/AstraStack/share/locale/
    cp -r share/themes dist/AstraStack/share/
    cp -r share/icons dist/AstraStack/share/
    cp -r ui dist/AstraStack/
    C:/'Program Files (x86)'/'Inno Setup 6'/ISCC.exe scripts/inno.iss
fi

