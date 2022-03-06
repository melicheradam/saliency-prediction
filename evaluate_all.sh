#!/bin/bash

# PSD Dataset
GENERATED_GENERALIZED_MAPS_PATH='data/PSD/orig_generalized'
GENERATED_PERSONALIZED_MAPS_PATH='data/PSD/fixations/Personalized_maps'
ORIGINAL_IMAGE_STIMULI_PATH='data/PSD/images'
ORIGINAL_PERSONALIZED_MAPS_PATH='data/PSD/fix_orig'

# Alzheimer dataset
#GENERATED_GENERALIZED_MAPS_PATH='eyetrackingdata/alzheimer-dataset-2/Generated_Personalized_Maps/Con/SM'
#GENERATED_PERSONALIZED_MAPS_PATH='eyetrackingdata/alzheimer-dataset-2/Generated_Personalized_Maps'
#ORIGINAL_IMAGE_STIMULI_PATH='eyetrackingdata/alzheimer-dataset-2/Slides'

rm -rf ./training_logs/
mkdir training_logs

declare -a OBSERVER_NAMES=($(ls $GENERATED_PERSONALIZED_MAPS_PATH))

: << 'COMMENT'
echo "Loading salicon model..."
encoder-decoder-model/load_model.sh salicon

## Reset the model into the SALICON state
echo "Removing all data from previous personalized training..."
sudo rm -f $(find encoder-decoder-model/results/history/ -type f -name '*personalized*')
sudo rm -f $(find encoder-decoder-model/results/ckpts/ -type f -name '*personalized*')

## Fine-tune the generalized model on the generalized maps from the personalized dataset

# Copy all stimuli files and generalized saliency maps
echo "Copying stimuli files..."
sudo rm -rf encoder-decoder-model/data/personalized/stimuli
sudo rm -rf encoder-decoder-model/data/personalized/val-stimuli
cp -r $ORIGINAL_IMAGE_STIMULI_PATH encoder-decoder-model/data/personalized/stimuli

echo "Copying generalized saliency maps..."
sudo rm -rf encoder-decoder-model/data/personalized/saliency
sudo rm -rf encoder-decoder-model/data/personalized/val-saliency
cp -r $GENERATED_GENERALIZED_MAPS_PATH encoder-decoder-model/data/personalized/saliency

# Train the generalized model
echo "Training the generalized model on generalized saliency maps from the personalized dataset..."
sudo docker run -u root -v $(pwd):/labs -it --gpus all python3-tensorflow-dp python encoder-decoder-model/main.py train -d personalized

## Serialize the generalized model in case it is needed in future
echo "Serializing the generalized model..."
sudo bash encoder-decoder-model/serialize_current_model.sh generalized
COMMENT
# Proceed with personalized training
i=1
for observer in "${OBSERVER_NAMES[@]}";do

  OBSERVER_NAME=$observer
  mkdir training_logs/"$OBSERVER_NAME"

  echo "----------------------------------------------------------------------------------------------------------------"
  echo "STARTING EXPERIMENT FOR OBSERVER named $observer ($i/${#OBSERVER_NAMES[@]})"

  ## Restore the model back to the generalized state to enable its training for other observers
  echo "Restoring model back to the generalized state..."
  sudo bash encoder-decoder-model/load_model.sh generalized

: << 'COMMENT'
  ## Predict saliency maps for the validation image set using the generalized model
  echo "Predicting saliency maps using the generalized model..."
  sudo docker run -u root -v $(pwd):/labs -it --gpus all python3-tensorflow-dp python encoder-decoder-model/main.py test -d personalized -p encoder-decoder-model/data/personalized/test
  cp -r encoder-decoder-model/results/images training_logs/"$OBSERVER_NAME"
  mv training_logs/"$OBSERVER_NAME"/images training_logs/"$OBSERVER_NAME"/generalized-saliency

  ## Produce discrepancy maps for the observer's predictions
  echo "Producing discrepancy maps for the generalized model and the observer..."
  rm -rf training_logs/"$OBSERVER_NAME"/discrepancy-generalized
  mkdir training_logs/"$OBSERVER_NAME"/discrepancy-generalized
  sudo docker run -u root -v $(pwd):/labs -it python3-tensorflow-dp python src/differentiate_maps.py -gt data/PSD/fix_orig/"$OBSERVER_NAME" -sal encoder-decoder-model/results/images -out training_logs/"$OBSERVER_NAME"/discrepancy-generalized -orig encoder-decoder-model/data/personalized/test

  ## Evaluate the performance of generalized model for a chosen observer
  echo "Evaluating performance of the generalized model..."
  sudo docker run -u root -v $(pwd):/labs -it python2-dp python src/evaluate_results.py -gt data/PSD/fix_orig/"$OBSERVER_NAME" -raw $GENERATED_PERSONALIZED_MAPS_PATH/$OBSERVER_NAME/FM -sal encoder-decoder-model/results/images -output ./training_logs/"$OBSERVER_NAME"/generalized.json -pg training_logs/"$OBSERVER_NAME"/generalized-saliency
COMMENT
  ## Train the personalized model
  echo "Training the personalized model..."
  sudo bash evaluate_single.sh "$observer" | tee ./training_logs/"$observer"/log.txts


  i=$(expr $i + 1)
done

# Print results
echo "Training complete ! Overall results:"
sudo docker run -u root -v $(pwd):/labs -it --gpus all python3-dp python src/show_training_results.py
