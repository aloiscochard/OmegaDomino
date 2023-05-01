#!/usr/bin/env bash

cd ./nnets

rm *.pb* > /dev/null

for file in *.py
do
  python $file EVAL&
  python $file PREDICT&
  python $file TRAIN
done


