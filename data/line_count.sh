#!/bin/sh

for i in `seq 0 113`; do
  wc -l ./graph/$i.csv | cut -f1 -d ' '
done
