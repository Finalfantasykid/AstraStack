#!/bin/bash

# Get location of libxcb.so.1 so it can be included
LIBXCB=`ldconfig -p | grep libxcb.so.1 | grep x86-64 | sed 's/^.*=> //'`

rm -fr dist
pyinstaller AstraStack.py --add-binary "$LIBXCB:."
rm -fr dist/AstraStack/share/icons/
rm -fr dist/AstraStack/share/themes/
cp -r share/themes dist/AstraStack/share/
cp -r share/icons dist/AstraStack/share/
cp -r ui dist/
cp bin/astra-stack dist/
cp bin/astra-stack.desktop dist/
cd dist
tar cfz ../AstraStack.tar.gz *
