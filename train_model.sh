#!/bin/bash
###
#
#

MODEL="nothing"
MODEL_NAME="nothing"
OBSERVER="nothing"

function parse_args(){
  while getopts "h?tno:" opt; do
    case "$opt" in
      h|\?)
        echo "Available arguments:"
        echo "  -t <model_type>       generalized, personalized"
        echo "  -n <model_name>       name of the model which will be serialized"
        echo "  -o <observer_name>    name of observer"
        exit 0
        ;;
      t)  MODEL=$OPTARG
        ;;
      n)  MODEL_NAME=$OPTARG
        ;;
      o)  OBSERVER=$OPTARG
        ;;
    esac
  done

  # if user does not set model name set it same as model type
  if [ "$MODEL_NAME" = "nothing" ]
  then
    MODEL_NAME=$MODEL
  fi
}

# PSD Dataset
IMAGE_PATH='data/PSD/images'

parse_args

if [ "$MODEL" = "generalized" ]
then
  GT_PATH='data/PSD/orig_generalized'
  echo "$MODEL"
elif [ "$MODEL" = "personalized" ]
then
  GT_PATH='data/PSD/fix_orig'
  echo "$MODEL"
else
  echo "Incorrect model type, valid options are: generalized, personalized"
  echo "Exiting..."
  exit 0
fi


function train_generalized(){
  echo "Loading salicon model..."
  encoder-decoder-model/load_model.sh salicon

  # Copy all stimuli files and generalized saliency maps
  echo "Copying stimuli files..."
  sudo rm -rf encoder-decoder-model/data/personalized/stimuli
  sudo rm -rf encoder-decoder-model/data/personalized/val-stimuli
  cp -r $IMAGE_PATH encoder-decoder-model/data/personalized/stimuli

  echo "Copying generalized saliency maps..."
  sudo rm -rf encoder-decoder-model/data/personalized/saliency
  sudo rm -rf encoder-decoder-model/data/personalized/val-saliency
  cp -r $GT_PATH encoder-decoder-model/data/personalized/saliency

  # Train the generalized model
  echo "Training the generalized model on generalized saliency maps from the personalized dataset..."
  sudo docker run -u root -v $(pwd):/labs -it --gpus all python3-tensorflow-dp python encoder-decoder-model/main.py train -d personalized

  ## Serialize the generalized model in case it is needed in future
  echo "Serializing the generalized model..."
  sudo bash encoder-decoder-model/serialize_current_model.sh "$MODEL_NAME"
}

function train_personalized(){
  # Copy all stimuli files
  echo "Copying stimuli files..."
  sudo rm -rf encoder-decoder-model/data/personalized/stimuli
  sudo rm -rf encoder-decoder-model/data/personalized/val-stimuli
  cp -r $IMAGE_PATH encoder-decoder-model/data/personalized/stimuli

  # Train a model personalized for a chosen observer
  echo "Copying saliency maps..."
  sudo rm -rf encoder-decoder-model/data/personalized/saliency
  sudo rm -rf encoder-decoder-model/data/personalized/val-saliency
  cp -r $GT_PATH/$OBSERVER encoder-decoder-model/data/personalized/saliency

  echo "Training personalized model..."
  sudo docker run -u root -v $(pwd):/labs -it --gpus all python3-tensorflow-dp python encoder-decoder-model/main.py train -d personalized

  ## Serialize the generalized model in case it is needed in future
  echo "Serializing the personalized model..."
  sudo bash encoder-decoder-model/serialize_current_model.sh "$MODEL_NAME""_""$OBSERVER"
}









