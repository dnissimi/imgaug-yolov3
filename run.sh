#!/bin/bash

#source activate cuda10

for filename in *.png *.jpg; do
    [ -e "$filename" ] || continue
    # ... rest of the loop body
    python imgaug-yolov3.py 200 $filename
done
