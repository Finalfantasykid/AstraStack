#!/bin/bash

echo "Installing..."
rm -fr /usr/lib/astrastack || true
mkdir -p /usr/lib/astrastack
cp -r AstraStack/* /usr/lib/astrastack
cp AstraStack/astrastack /usr/bin/astrastack
cp AstraStack/astrastack.desktop /usr/share/applications/
ln -s /usr/share/ /usr/lib/astrastack/
