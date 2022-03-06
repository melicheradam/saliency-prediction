#!/bin/bash

declare -a OBSERVER_NAMES=($(ls data/PSD/fix_orig))

i=1
for observer in "${OBSERVER_NAMES[@]}";do
  echo "------- $observer --------"
  sudo bash train_model.sh -t personalized -n run1 -o $observer

  i=$(expr $i + 1)
done
