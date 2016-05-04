#!/bin/sh

javac -cp ../../lib/*.jar *.java

if [ ! -d "./log" ]; then
  mkdir -p log
fi

# sigma: try 2^{-4,-3,-2,-1,0,1,2,3,4}
# lambda: try 0 0.2 0.4 0.6 0.8
for seed in `seq 0 4`; do 
#  for sigma in 0.5 0.25 2 4; do
#  for lambda in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  for delta in 0.05 0.2 0.4 0.8; do
    nohup java -cp .:../../lib/Jama-1.0.3.jar Main $seed $delta > "./log/"$seed"_"$delta".txt" & 
  done
done

