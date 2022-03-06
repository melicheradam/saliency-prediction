#!/bin/bash
###
#
#

MODEL="nothing"
MODEL_NAME="nothing"
OBSERVER="nothing"

while getopts "t:n:o:" opt; do
  case "$opt" in
    t)  MODEL=$OPTARG
      ;;
    n)  MODEL_NAME=$OPTARG
      ;;
    o)  OBSERVER=$OPTARG
      ;;
    ?)
      echo "Available arguments:"
      echo "  -t <model_type>       generalized, personalized"
      echo "  -n <model_name>       name of the model which will be serialized"
      echo "  -o <observer_name>    name of observer"
      exit 0
      ;;
  esac
done

# if user does not set model name set it same as model type
if [ "$MODEL_NAME" = "nothing" ]
then
  MODEL_NAME=$MODEL
fi

if [ "$MODEL" = "generalized" ]
then
  GT_PATH='data/PSD/orig_generalized'
  echo "Loading model named ..."
  encoder-decoder-model/load_model.sh "$MODEL_NAME"
elif [ "$MODEL" = "personalized" ]
then
  GT_PATH='data/PSD/fix_orig'
  echo "Loading model named $MODEL_NAME""_""$OBSERVER..."
  encoder-decoder-model/load_model.sh "$MODEL_NAME""_""$OBSERVER"
else
  echo "Incorrect model type, valid options are: generalized, personalized"
  echo "Exiting..."
  exit 0
fi

echo "Predicting maps by $MODEL_NAME model..."
rm -rf test-results/"$MODEL_NAME"/"$OBSERVER"
mkdir -p test-results/"$MODEL_NAME"/"$OBSERVER"/"$MODEL"-discrepancy
mkdir -p test-results/"$MODEL_NAME"/"$OBSERVER"/"$MODEL"-saliency

## Predict saliency maps for the test image
sudo docker run -u root -v $(pwd):/labs -it --gpus all python3-tensorflow-dp python encoder-decoder-model/main.py test -d personalized -p encoder-decoder-model/data/personalized/test
cp -rf encoder-decoder-model/results/images test-results/"$MODEL_NAME"/"$OBSERVER"/"$MODEL"-saliency

## Evaluate all saliency metrics
echo "Evaluating performance..."
sudo docker run -u root -v $(pwd):/labs -it python2-dp python src/evaluate_results.py -gt $GT_PATH/"$OBSERVER" -raw $GT_PATH/"$OBSERVER" -sal test-results/"$MODEL_NAME"/"$OBSERVER"/"$MODEL"-saliency -output  test-results/"$MODEL_NAME"/"$OBSERVER"/personalized.json -pg test-results/"$MODEL_NAME"/"$OBSERVER"/"$MODEL"-saliency

## Produce discrepancy maps for the observer's predictions
echo "Producing discrepancy maps..."
sudo docker run -u root -v $(pwd):/labs -it python3-tensorflow-dp python src/differentiate_maps.py -gt $GT_PATH/"$OBSERVER" -sal test-results/"$MODEL_NAME"/"$OBSERVER"/saliency -out test-results/"$MODEL_NAME"/"$OBSERVER"/"$MODEL"-discrepancy -orig encoder-decoder-model/data/personalized/test

