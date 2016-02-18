#!/bin/sh

for i in `seq 0 1000`; do
  if [ ! -d "./save/$i" ]; then
    mkdir ./save/"$i"
  fi
done

