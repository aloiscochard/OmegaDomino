#!/usr/bin/env bash

cd ./graphs

rm *.pb* > /dev/null

for file in *.py
do
  python $file
done


