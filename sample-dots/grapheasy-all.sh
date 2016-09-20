#!/bin/bash

for file in *.dot; do
    echo $file
    graph-easy $file
done
