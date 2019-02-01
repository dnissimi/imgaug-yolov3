#!/bin/bash

source activate cuda10

for filename in ../origs/*.jpg; do
    [ -e "$filename" ] || continue
    # ... rest of the loop body
    python ../../aug_img+bb.py 50 $filename
done
