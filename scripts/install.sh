#!/bin/bash

if [ "$1" == "" ] || [ "$1" == "help" ]; then
    echo "Usage:"
    echo "    sudo ./install.sh install"
    echo "    sudo ./install.sh remove"
elif [ `whoami` != "root" ]; then
    echo "You must run this script as root (try sudo)"
elif [ "$1" == "install" ]; then
    echo "Installing..."
    rm -fr /usr/lib/astrastack || true
    mkdir -p /usr/lib/astrastack
    cp -r AstraStack/* /usr/lib/astrastack
    cp AstraStack/astrastack /usr/bin/astrastack
    cp AstraStack/astrastack.desktop /usr/share/applications/
    ln -s /usr/share/ /usr/lib/astrastack/
elif [ "$1" == "remove" ]; then
    echo "Removing..."
    rm -fr /usr/lib/astrastack || true
    rm -fr /usr/bin/astrastack || true
    rm -fr usr/share/applications/astrastack.desktop || true
fi
