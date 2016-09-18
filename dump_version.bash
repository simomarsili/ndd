#!/bin/bash 

NEW_VERSION=$1
source VERSION

if [ -z "$NEW_VERSION" ]; then
    echo "a flag for the new version is needed, e.g.: dump_version.bash v1.0.3"
    exit
fi

for file in VERSION ndd.py; do 
    sed "s/$VERSION/$NEW_VERSION/g" $file > tmp; 
    mv tmp $file
done
