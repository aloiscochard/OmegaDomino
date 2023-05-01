#!/usr/bin/env bash
# cleanup.sh <directory> <epoch_to_keep>
# cleanup.sh ./ 2

epochs=$(ls -d ${1%/}/epoch-*)
epochs_len=$(ls -d ${1%/}/epoch-* | wc -w)

echo "Processing $epochs_len epochs ..."
i=0;
n=$2;
for epoch in $epochs; do
  i=$((i+1))
  if (( i > (epochs_len-n) )); then
    echo "$epoch: keep"
  else
    echo "$epoch: removing"
    rm -rf $epoch
  fi
done


