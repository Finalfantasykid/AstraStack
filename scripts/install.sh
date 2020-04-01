#!/bin/bash

echo "Installing..."
mkdir -p /usr/lib/astrastack
cp -r AstraStack/* /usr/lib/astrastack
cp AstraStack/astrastack /usr/bin/astrastack
cp AstraStack/astrastack.desktop /usr/share/applications/
