#!/usr/bin/env bash

set -e

# CFG=--config=opt
CFG="--config=opt --config=cuda --copt=-msse4.1 --copt=-msse4.2"

AR=$(which ar)
AR_PATTERN="s,/usr/bin/ar,${AR},g"

virtualenv build
source build/bin/activate

pip install enum34 mock numpy six

./configure

find -type f -name CROSSTOOL\* -exec sed -i \
  -e $AR_PATTERN \
  {} \;

bazel build $CFG //tensorflow/tools/pip_package:build_pip_package 
bazel build $CFG //tensorflow:libtensorflow.so

sed -i 's,.*bdist_wheel.*,cp -rL . "${DEST}"; exit 0,' bazel-bin/tensorflow/tools/pip_package/build_pip_package 

rm -rf ~/.local/tensorflow/
mkdir ~/.local/tensorflow/

bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/.local/tensorflow
cp ./bazel-bin/tensorflow/libtensorflow*.so ~/.local/tensorflow
