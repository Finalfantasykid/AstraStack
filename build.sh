#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    # Get location of libxcb.so.1 so it can be included
    LIBXCB=`ldconfig -p | grep libxcb.so.1 | grep x86-64 | sed 's/^.*=> //'`

    rm -fr dist
    pyinstaller --add-binary "$LIBXCB:." \
                --hidden-import "cairo" \
                --exclude-module "gi.repository.Gst" \
                AstraStack.py
    rm -fr dist/AstraStack/*.dist-info
    rm -fr dist/AstraStack/share/
    cp -r ui dist/AstraStack/
    cp -r manual dist/AstraStack/
    rm dist/AstraStack/ui/logo.xcf
    rm dist/AstraStack/ui/logo.ico
    rm dist/AstraStack/manual/Manual.odt
    cp scripts/astrastack dist/AstraStack/
    cp scripts/astrastack.desktop dist/AstraStack
    
    if [[ "$1" != "snap" ]]; then
        # Normal Linux build
        mkdir -p dist/AstraStack/share_override/icons/Adwaita/
        cp -r share_override/icons/Adwaita dist/AstraStack/share_override/icons/
        cp scripts/install.sh dist/
        cd dist
        echo "Compressing..."
        tar cf - . -P | pv -s $(du -sb . | awk '{print $1}') | xz > ../AstraStack.tar.xz
    else
        # Snap build
        cp scripts/astrastack.snap.desktop dist/AstraStack/ui/AstraStack.desktop
        snapcraft
    fi
elif [[ "$OSTYPE" == "msys" ]]; then
    rm -fr dist
    pyinstaller --windowed \
                --icon "ui/logo.ico" \
                --hidden-import "packaging.requirements" \
                --hidden-import "pkg_resources.py2_warn" \
                --hidden-import "cairo" \
                --exclude-module "FixTk" \
                --exclude-module "tcl" \
                --exclude-module "tk" \
                --exclude-module "_tkinter" \
                --exclude-module "tkinter" \
                --exclude-module "Tkinter" \
                AstraStack.py
    rm -fr dist/AstraStack/*.dist-info
    rm -fr dist/AstraStack/share/locale/
    rm -fr dist/AstraStack/site-packages/
    cp -r ui dist/AstraStack/
    cp -r manual dist/AstraStack/
    rm dist/AstraStack/ui/logo.xcf
    rm dist/AstraStack/ui/logo.ico
    rm dist/AstraStack/manual/Manual.odt
    C:/'Program Files (x86)'/'Inno Setup 6'/ISCC.exe scripts/inno.iss
fi
