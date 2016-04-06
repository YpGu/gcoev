#!/bin/sh
# $1 is lambda
for i in 0.0125 0.025 0.05 0.1 0.2 0.4 0.8 1.6; do python classification.py ./all_save/$1/$i/ | awk '{sum+=$2; n++} END {if(n>0) print sum/n}'; done

#for i in 0.0125 0.025 0.05 0.1 0.2 0.4 0.8 1.6; do python classification.py ./all_save/em/$i/ | awk '{sum+=$2; n++} END {if(n>0) print sum/n}'; done

