#!/usr/bin/env bash

# MNIST
declare -a mnist_files=("train-images-idx3" "train-labels-idx1" "t10k-images-idx3" "t10k-labels-idx1")
mkdir -p ./datasets/mnist
for file in "${mnist_files[@]}"
do
  curl  --output "./datasets/mnist/${file}.gz" -O "http://yann.lecun.com/exdb/mnist/${file}-ubyte.gz"
  gzip -d ./datasets/mnist/${file}.gz
done

