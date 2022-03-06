#!/bin/bash

# PSD Dataset
GENERATED_GENERALIZED_MAPS_PATH='data/PSD/orig_generalized'
GENERATED_PERSONALIZED_MAPS_PATH='data/PSD/fixations/Personalized_maps'
ORIGINAL_IMAGE_STIMULI_PATH='data/PSD/images'
ORIGINAL_PERSONALIZED_MAPS_PATH='data/PSD/fix_orig'

# Alzheimer dataset
#GENERATED_GENERALIZED_MAPS_PATH='eyetrackingdata/alzheimer-dataset-2/Generated_Generalized_maps'
#GENERATED_PERSONALIZED_MAPS_PATH='eyetrackingdata/alzheimer-dataset-2/Generated_Personalized_Maps'
#ORIGINAL_IMAGE_STIMULI_PATH='eyetrackingdata/alzheimer-dataset-2/Slides'

OBSERVER_NAME=$1

# Copy all stimuli files
echo "Copying stimuli files..."
sudo rm -rf encoder-decoder-model/data/personalized/stimuli
sudo rm -rf encoder-decoder-model/data/personalized/val-stimuli
cp -r $ORIGINAL_IMAGE_STIMULI_PATH encoder-decoder-model/data/personalized/stimuli

# Copy only saliency maps which are available for the given observer so that we can later evaluate the results on the same data
#python3 src/prepareData.py -genpath $GENERATED_GENERALIZED_MAPS_PATH -perspath $GENERATED_PERSONALIZED_MAPS_PATH/$OBSERVER_NAME/SM -targetpath encoder-decoder-model/data/personalized/saliency

# Train a model personalized for a chosen observer
echo "Preparing data for personalized training..."
sudo rm -rf encoder-decoder-model/data/personalized/saliency
sudo rm -rf encoder-decoder-model/data/personalized/val-saliency
cp -r data/PSD/fix_orig/$OBSERVER_NAME encoder-decoder-model/data/personalized/saliency
sudo docker run -u root -v $(pwd):/labs -it --gpus all python3-tensorflow-dp python encoder-decoder-model/main.py train -d personalized

## Predict saliency maps for the test image set using the personalized model
echo "Predicting maps by the personalized model..."
sudo docker run -u root -v $(pwd):/labs -it --gpus all python3-tensorflow-dp python encoder-decoder-model/main.py test -d personalized -p encoder-decoder-model/data/personalized/test
cp -r encoder-decoder-model/results/images training_logs/"$OBSERVER_NAME"
mv training_logs/"$OBSERVER_NAME"/images training_logs/"$OBSERVER_NAME"/personalized-saliency

## Evaluate all saliency metrics
echo "Evaluating performance of the personalized model..."
sudo docker run -u root -v $(pwd):/labs -it python2-dp python src/evaluate_results.py -gt data/PSD/fix_orig/"$OBSERVER_NAME" -raw $GENERATED_PERSONALIZED_MAPS_PATH/$OBSERVER_NAME/FM -sal encoder-decoder-model/results/images -output ./training_logs/"$OBSERVER_NAME"/personalized.json -pg training_logs/"$OBSERVER_NAME"/personalized-saliency

## Produce discrepancy maps for the observer's predictions
echo "Producing discrepancy maps..."
rm -rf training_logs/"$OBSERVER_NAME"/discrepancy-personalized
mkdir training_logs/"$OBSERVER_NAME"/discrepancy-personalized
sudo docker run -u root -v $(pwd):/labs -it python3-tensorflow-dp python src/differentiate_maps.py -gt data/PSD/fix_orig/"$OBSERVER_NAME" -sal encoder-decoder-model/results/images -out training_logs/"$OBSERVER_NAME"/discrepancy-personalized -orig encoder-decoder-model/data/personalized/test

## Store weights from the last encoder layer
#sudo docker run -u root -v $(pwd):/labs -it --gpus all python3-tensorflow-dp python encoder-decoder-model/parse_weights.py test -d personalized -p encoder-decoder-model/data/personalized/test -o training_logs/$OBSERVER_NAME

## Compare the performance of the personalized model with Unisal generalized model
#echo "Predicting saliency maps with Unisal Generalized model..."
#rm -rf unisal/examples/personalized
#mkdir unisal/examples/personalized
#cp -r encoder-decoder-model/data/personalized/test unisal/examples/personalized
#mv unisal/examples/personalized/test unisal/examples/personalized/images

#source /home/tom/miniconda3/etc/profile.d/conda.sh
#conda activate unisal
#python unisal/run.py predict_examples
#conda deactivate

#cp -r unisal/examples/personalized/saliency training_logs/"$OBSERVER_NAME"
#mv training_logs/"$OBSERVER_NAME"/saliency training_logs/"$OBSERVER_NAME"/unisal-saliency

#echo "Evaluating performance of the Unisal Generalized model..."
#sudo docker run -u root -v $(pwd):/labs -it python2-dp python src/evaluate_results.py -gt $GENERATED_PERSONALIZED_MAPS_PATH/"$OBSERVER_NAME"/SM -raw $GENERATED_PERSONALIZED_MAPS_PATH/$OBSERVER_NAME/FM -sal unisal/examples/personalized/saliency -output ./training_logs/"$OBSERVER_NAME"/unisal.json -pg training_logs/"$OBSERVER_NAME"/generalized-saliency

# Produce discrepancy maps for Unisal predictions
#echo "Producing discrepancy maps..."
#rm -rf training_logs/"$OBSERVER_NAME"/discrepancy-unisal
#mkdir training_logs/"$OBSERVER_NAME"/discrepancy-unisal
#sudo docker run -u root -v $(pwd):/labs -it python3-tensorflow-dp python src/differentiate_maps.py -gt $GENERATED_PERSONALIZED_MAPS_PATH/"$OBSERVER_NAME"/SM -sal unisal/examples/personalized/saliency -out training_logs/"$OBSERVER_NAME"/discrepancy-unisal -orig encoder-decoder-model/data/personalized/test

