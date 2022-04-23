#!/bin/sh
docker run -u root -v "$(pwd)":/labs -it python2-dp python src/evaluate_results.py -gt data/PSD/fix_orig/"$OBSERVER_NAME" -raw $GENERATED_PERSONALIZED_MAPS_PATH/$OBSERVER_NAME/FM -sal encoder-decoder-model/results/images -output ./training_logs/"$OBSERVER_NAME"/personalized.json -pg training_logs/"$OBSERVER_NAME"/personalized-saliency


sudo docker run -v "$PWD:/workdir" -u "$(id -u):$(id -g)" \
  --rm -it dhuser/dhimage dhprog "$@"