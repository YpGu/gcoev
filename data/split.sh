#!/bin/sh

#dir="./jsim_graph_selected/"
dir="./nips/"

for i in `seq 0 115`; do
  if [ -e $dir$i.csv ]; then 
    printf "%d, " $i
    awk 'FNR%10!=0 {print $0}' $dir$i.csv > $dir$i.train.csv
    awk 'FNR%10==0 {print $0}' $dir$i.csv > $dir$i.test.csv
    awk 'FNR%10!=0 {print $0}' "$dir""$i"_neg.csv > "$dir""$i"_neg.train.csv
    awk 'FNR%10==0 {print $0}' "$dir""$i"_neg.csv > "$dir""$i"_neg.test.csv
  else
    break
  fi
done

echo 

