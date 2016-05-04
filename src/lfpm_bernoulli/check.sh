#!/bin/sh
sigma=$1
for i in `seq 0 4`; do 
  python link_prediction.py $i $sigma
#done | awk '{sum+=$1; x+=1} END {print sum/x}'
done | awk '{a[i++]=$1} END {for (i in a) {mean+=a[i]; x+=1} mean/=x; for (i in a) var+=(a[i]-mean)*(a[i]-mean); var/=x; print mean, sqrt(var)}'
