# Sentiment Cartography

## Our method
### Train positive & negative models
envs/bin/python3 train.py
### ... and infer
envs/bin/python3 infer.py
### ... and infer with augmented dataset
envs/bin/python3 infer_augment.py

## Classification baselines
### Multiclass
envs/bin/python3 00_baseline_sequence_classification.py
### Binary
envs/bin/python3 00_baseline_sequence_classification.py --num_labels 2
