#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    # Get location of libxcb.so.1 so it can be included
    LIBXCB=`ldconfig -p | grep libxcb.so.1 | grep x86-64 | sed 's/^.*=> //'`

    rm -fr dist
    pyinstaller --add-binary "$LIBXCB:." \
                --hidden-import "cairo" \
                AstraStack.py
    rm -fr dist/AstraStack/share/
    mkdir -p dist/AstraStack/share_override/icons/Adwaita/
    cp -r share_override/icons/Adwaita dist/AstraStack/share_override/icons/
    cp -r ui dist/AstraStack/
    cp -r manual dist/AstraStack/
    rm dist/AstraStack/ui/logo.xcf
    rm dist/AstraStack/ui/logo.ico
    rm dist/AstraStack/manual/Manual.odt
    cp scripts/astrastack dist/AstraStack/
    cp scripts/astrastack.desktop dist/AstraStack
    cp scripts/install.sh dist/
    cd dist
    tar cjf ../AstraStack.tar.bz2 *
elif [[ "$OSTYPE" == "msys" ]]; then
    rm -fr dist
    pyinstaller --windowed \
                --icon "ui/logo.ico" \
                --hidden-import "packaging.requirements" \
                --hidden-import "pkg_resources.py2_warn" \
                --hidden-import "cairo" \
                --add-binary "/mingw64/bin/opencv_videoio_ffmpeg420_64.dll:." \
                --exclude-module "FixTk" \
                --exclude-module "tcl" \
                --exclude-module "tk" \
                --exclude-module "_tkinter" \
                --exclude-module "tkinter" \
                --exclude-module "Tkinter" \
                AstraStack.py
    rm -fr dist/AstraStack/share/locale/
    cp -r ui dist/AstraStack/
    cp -r manual dist/AstraStack/
    rm dist/AstraStack/ui/logo.xcf
    rm dist/AstraStack/ui/logo.ico
    rm dist/AstraStack/manual/Manual.odt
    C:/'Program Files (x86)'/'Inno Setup 6'/ISCC.exe scripts/inno.iss
fi
