#!/bin/bash

#source activate cuda10

for filename in *.jpg; do
    [ -e "$filename" ] || continue
    # ... rest of the loop body
    python imgaug-yolov3.py 50 $filename
done
