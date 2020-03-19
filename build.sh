#!/bin/bash

rm -fr dist
pyinstaller AstraStack.py
rm -fr dist/AstraStack/share/icons/
rm -fr dist/AstraStack/share/themes/
cp -r share/themes dist/AstraStack/share/
cp -r share/icons dist/AstraStack/share/
cp -r ui dist/
cp bin/astra-stack dist/
cp bin/astra-stack.desktop dist/
cd dist
tar cfz ../AstraStack.tar.gz *
