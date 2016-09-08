#!/bin/bash 

NEW_VERSION=$1
source VERSION

for file in VERSION ndd.py README.md; do 
    sed "s/$VERSION/$NEW_VERSION/g" $file > tmp; 
    mv tmp $file
done
