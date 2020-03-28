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
    cp bin/astra-stack dist/
    cp bin/astra-stack.desktop dist/
    cd dist
    tar cfz ../AstraStack.tar.gz *
elif [[ "$OSTYPE" == "msys" ]]; then
    rm -fr dist
    rm AstraStack.zip
    pyinstaller --windowed \
                --icon "ui/logo.ico" \
                --hidden-import "packaging.requirements" \
                --hidden-import "pkg_resources.py2_warn" \
                --add-binary "/mingw64/bin/opencv_videoio_ffmpeg420_64.dll:." \
                AstraStack.py
    cp -r ui dist/
    cd dist
    zip -r ../AstraStack *
fi
