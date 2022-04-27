# Preprocess the Personalized dataset

# Genarate generalized saliency maps from fixation files
# Default repository configuration:
# - Use Gauss Kernel of 303x303 to blur the saliency maps
# - Do not merge close fixations

sudo docker run -u root -v $(pwd):/labs -it python3-tensorflow-dp python src/fixToGeneralizedMap.py -fix data/PSD/fixations/ -out generated/Generated_Generalized_maps/ && sudo chown -R "$(id -u):$(id -g)" .  

# Generate personalized saliency maps from fixation files
# Default repository configuration:
# - Use Gauss Kernel of 303x303 to blur the saliency maps
# - Do not merge close fixations

sudo docker run -u root -v $(pwd):/labs -it python3-tensorflow-dp python src/fixToSaliencyMap.py -fix data/PSD/fixations/ && sudo chown -R "$(id -u):$(id -g)" .

# Generate generalized heatmaps to visualize fixations over an image
sudo docker run -u root -v $(pwd):/labs -it python3-tensorflow-dp python src/fixToGeneralizedAlphaHeatmap.py -fix generated/Generated_Personalized_maps -out data/Generated_Generalized_heatmaps -orig data/PSD/images

