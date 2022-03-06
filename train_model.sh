#!/bin/bash
###
#
#

# PSD Dataset
GENERATED_GENERALIZED_MAPS_PATH='data/PSD/orig_generalized'
GENERATED_PERSONALIZED_MAPS_PATH='data/PSD/fixations/Personalized_maps'
ORIGINAL_IMAGE_STIMULI_PATH='data/PSD/images'
ORIGINAL_PERSONALIZED_MAPS_PATH='data/PSD/fix_orig'

MODEL=$1

if [ "$MODEL" = "generalized" ]
then
  SAL =
  echo "$MODEL"
elif [ "$MODEL" = "personalized" ]
then
  echo "$MODEL"
else
  echo "Incorrect arguments, valid options are: generalized, personalized"
  echo "Exiting..."
  exit 0
fi


