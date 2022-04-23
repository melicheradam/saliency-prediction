GENERATED_GENERALIZED_MAPS_PATH='data/PSD/orig_generalized'
GENERATED_PERSONALIZED_MAPS_PATH='data/PSD/fixations/Personalized_maps'
ORIGINAL_IMAGE_STIMULI_PATH='data/PSD/images'

# Train the model on SALICON dataset
#sudo docker run -u root -v $(pwd):/labs -it --gpus all python3-tensorflow-dp python encoder-decoder-model/main.py train -d salicon -p encoder-decoder-model/data/salicon/
# load salicon model

declare -a OBSERVER_NAMES=($(ls data/PSD/fixations/Personalized_maps))

echo "Loading salicon model..."
# encoder-decoder-model/load_model.sh salicon

## Reset the model into the SALICON state
echo "Removing all data from previous personalized training..."
sudo rm -f $(find encoder-decoder-model/results/history/ -type f -name '*personalized*')
sudo rm -f $(find encoder-decoder-model/results/ckpts/ -type f -name '*personalized*')


# Copy all stimuli files and generalized saliency maps
echo "Copying stimuli and generalized saliency files..."
sudo rm -rf encoder-decoder-model/data/personalized/saliency
sudo rm -rf encoder-decoder-model/data/personalized/val-saliency
sudo rm -rf encoder-decoder-model/data/personalized/stimuli
sudo rm -rf encoder-decoder-model/data/personalized/val-stimuli
cp -r $GENERATED_GENERALIZED_MAPS_PATH encoder-decoder-model/data/personalized/saliency
cp -r $ORIGINAL_IMAGE_STIMULI_PATH encoder-decoder-model/data/personalized/stimuli

# Train the generalized model
echo "Training the generalized model on generalized saliency maps from the personalized dataset..."
sudo docker run -u root -v $(pwd):/labs -it --gpus all python3-tensorflow-dp python encoder-decoder-model/main.py train -d personalized

## Serialize the generalized model in case it is needed in future
echo "Serializing the generalized model..."
sudo encoder-decoder-model/serialize_current_model.sh generalized

## Predict saliency maps for the validation image set using the generalized model
# echo "Predicting saliency maps using the generalized model..."
# sudo docker run -u root -v $(pwd):/labs -it --gpus all python3-tensorflow-dp python encoder-decoder-model/main.py test -d personalized -p encoder-decoder-model/data/personalized/val-stimuli
# cp -r encoder-decoder-model/results/images training_logs/
# mv training_logs/images training_logs/generalized-saliency

## Produce discrepancy maps for the observer's predictions
# echo "Producing discrepancy maps for the generalized model and the observer..."
# rm -rf training_logs/discrepancy-generalized
# mkdir training_logs/discrepancy-generalized
# sudo docker run -u root -v $(pwd):/labs -it python3-tensorflow-dp python src/differentiate_maps.py -gt data/PSD/orig_generalized -sal encoder-decoder-model/results/images -out training_logs/discrepancy-generalized -orig encoder-decoder-model/data/personalized/val-stimuli

## Evaluate the performance of generalized model for a chosen observer
# echo "Evaluating performance of the generalized model..."
# sudo docker run -u root -v $(pwd):/labs -it python2-dp python src/evaluate_results.py -gt data/PSD/orig_generalized -raw data/PSD/orig_generalized -sal encoder-decoder-model/results/images -output training_logs/generalized.json -pg encoder-decoder-model/results/images

# Proceed with personalized training
i=1
for observer in "${OBSERVER_NAMES[@]}";do

  OBSERVER_NAME=$observer
  mkdir training_logs/"$OBSERVER_NAME"

  echo "----------------------------------------------------------------------------------------------------------------"
  echo "STARTING EXPERIMENT FOR OBSERVER named $observer ($i/${#OBSERVER_NAMES[@]})"

  ## Predict saliency maps for the validation image set using the generalized model
  echo "Predicting saliency maps using the generalized model..."
  sudo docker run -u root -v $(pwd):/labs -it --gpus all python3-tensorflow-dp python encoder-decoder-model/main.py test -d personalized -p encoder-decoder-model/data/personalized/test
  cp -r encoder-decoder-model/results/images training_logs/"$OBSERVER_NAME"
  mv training_logs/"$OBSERVER_NAME"/images training_logs/"$OBSERVER_NAME"/generalized-saliency

  ## Produce discrepancy maps for the observer's predictions
  echo "Producing discrepancy maps for the generalized model and the observer..."
  rm -rf training_logs/"$OBSERVER_NAME"/discrepancy-generalized
  mkdir training_logs/"$OBSERVER_NAME"/discrepancy-generalized
  sudo docker run -u root -v $(pwd):/labs -it python3-tensorflow-dp python src/differentiate_maps.py -gt data/PSD/fix_orig/"$OBSERVER_NAME" -sal encoder-decoder-model/results/images -out training_logs/"$OBSERVER_NAME"/discrepancy-generalized -orig encoder-decoder-model/data/personalized/val-stimuli

  ## Evaluate the performance of generalized model for a chosen observer
  echo "Evaluating performance of the generalized model..."
  sudo docker run -u root -v $(pwd):/labs -it python2-dp python src/evaluate_results.py -gt data/PSD/fix_orig/"$OBSERVER_NAME" -raw $GENERATED_PERSONALIZED_MAPS_PATH/$OBSERVER_NAME/FM -sal encoder-decoder-model/results/images -output ./training_logs/"$OBSERVER_NAME"/generalized.json -pg training_logs/"$OBSERVER_NAME"/generalized-saliency

  ## Restore the model back to the generalized state to enable its training for other observers
  echo "Restoring model back to the generalized state..."
  sudo bash encoder-decoder-model/load_model.sh generalized

  i=$(expr $i + 1)
done

# Print results
echo "Training complete ! Overall results:"
sudo docker run -u root -v $(pwd):/labs -it --gpus all python3-dp python src/show_training_results.py

exit 0
## Train the personalized model
echo "--- Training the personalized model... ---"

# Copy all stimuli files
echo "Copying stimuli files..."
sudo rm -rf encoder-decoder-model/data/personalized/stimuli
sudo rm -rf encoder-decoder-model/data/personalized/val-stimuli
cp -r data/PSD/images encoder-decoder-model/data/personalized/stimuli

# Train a model personalized for a chosen observer
echo "Preparing data for personalized training..."
sudo rm -rf encoder-decoder-model/data/personalized/saliency
sudo rm -rf encoder-decoder-model/data/personalized/val-saliency
cp -r data/PSD/fixations/Personalized_maps/Sub_1/SM encoder-decoder-model/data/personalized/saliency

echo "Training the personalized model..."
sudo docker run -u root -v $(pwd):/labs -it --gpus all python3-tensorflow-dp python encoder-decoder-model/main.py train -d personalized

## Predict saliency maps for the test image set using the personalized model
echo "Predicting maps by the personalized model..."
sudo docker run -u root -v $(pwd):/labs -it --gpus all python3-tensorflow-dp python encoder-decoder-model/main.py test -d personalized -p encoder-decoder-model/data/personalized/val-stimuli
cp -r encoder-decoder-model/results/images training_logs/Sub_1
mv training_logs/Sub_1/images training_logs/Sub_1/personalized-saliency

## Evaluate all saliency metrics
echo "Evaluating performance of the personalized model..."
sudo docker run -u root -v $(pwd):/labs -it python2-dp python src/evaluate_results.py -gt data/PSD/fixations/Personalized_maps/Sub_1/SM -raw data/PSD/fixations/Personalized_maps/Sub_1/FM -sal encoder-decoder-model/results/images -output ./training_logs/Sub_1/personalized.json -pg training_logs/Sub_1/generalized-saliency

## Produce discrepancy maps for the observer's predictions
echo "Producing discrepancy maps..."
rm -rf training_logs/Sub_1/discrepancy-personalized
mkdir training_logs/Sub_1/discrepancy-personalized
sudo docker run -u root -v $(pwd):/labs -it python3-tensorflow-dp python src/differentiate_maps.py -gt data/PSD/fixations/Personalized_maps/Sub_1/SM -sal encoder-decoder-model/results/images -out training_logs/Sub_1/discrepancy-personalized -orig encoder-decoder-model/data/personalized/val-stimuli

## Restore the model back to the generalized state to enable its training for other observers
echo "Restoring model back to the generalized state..."
sudo bash encoder-decoder-model/load_model.sh generalized

# Print results
echo "Training complete ! Overall results:"
sudo docker run -u root -v $(pwd):/labs -it --gpus all python3-dp python src/show_training_results.py