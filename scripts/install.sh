#!/bin/bash

echo "Installing..."
cp -r AstraStack/* /usr/lib/astrastack
cp AstraStack/astrastack /usr/bin/astrastack
cp AstraStack/astrastack.desktop /usr/share/applications/
