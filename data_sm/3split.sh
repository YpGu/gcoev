#!/bin/sh

#dir="./jsim_graph_selected/"
#dir="./nips_17/out/"
#dir="./infocom/out/"
#dir="./covote/out_old/"
#dir="./synthetic/out/"
dir="./dblp_venue/"

T=50

for s in `seq 0 5`; do 
  if [ ! -d $dir$s ]; then
    mkdir -p $dir$s
  fi
done


for i in `seq 0 $T`; do
  if [ -e $dir$i.csv ]; then 
    printf "%d " $i
    awk 'FNR%5!=0 {print $0}' $dir$i.csv > $dir"0/"$i.train.csv
    awk 'FNR%5==0 {print $0}' $dir$i.csv > $dir"0/"$i.test.csv
    awk 'FNR%5!=0 {print $0}' $dir$i.neg.csv > $dir"0/"$i.train.neg.csv
    awk 'FNR%5==0 {print $0}' $dir$i.neg.csv > $dir"0/"$i.test.neg.csv

    awk 'FNR%5!=1 {print $0}' $dir$i.csv > $dir"1/"$i.train.csv
    awk 'FNR%5==1 {print $0}' $dir$i.csv > $dir"1/"$i.test.csv
    awk 'FNR%5!=1 {print $0}' $dir$i.neg.csv > $dir"1/"$i.train.neg.csv
    awk 'FNR%5==1 {print $0}' $dir$i.neg.csv > $dir"1/"$i.test.neg.csv

    awk 'FNR%5!=2 {print $0}' $dir$i.csv > $dir"2/"$i.train.csv
    awk 'FNR%5==2 {print $0}' $dir$i.csv > $dir"2/"$i.test.csv
    awk 'FNR%5!=2 {print $0}' $dir$i.neg.csv > $dir"2/"$i.train.neg.csv
    awk 'FNR%5==2 {print $0}' $dir$i.neg.csv > $dir"2/"$i.test.neg.csv

    awk 'FNR%5!=3 {print $0}' $dir$i.csv > $dir"3/"$i.train.csv
    awk 'FNR%5==3 {print $0}' $dir$i.csv > $dir"3/"$i.test.csv
    awk 'FNR%5!=3 {print $0}' $dir$i.neg.csv > $dir"3/"$i.train.neg.csv
    awk 'FNR%5==3 {print $0}' $dir$i.neg.csv > $dir"3/"$i.test.neg.csv

    awk 'FNR%5!=4 {print $0}' $dir$i.csv > $dir"4/"$i.train.csv
    awk 'FNR%5==4 {print $0}' $dir$i.csv > $dir"4/"$i.test.csv
    awk 'FNR%5!=4 {print $0}' $dir$i.neg.csv > $dir"4/"$i.train.neg.csv
    awk 'FNR%5==4 {print $0}' $dir$i.neg.csv > $dir"4/"$i.test.neg.csv
  else
    break
  fi
done

echo 

