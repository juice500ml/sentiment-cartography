# Sentiment Cartography

Install requirements using 
```
pip install -r requirements.txt
```
## Our method
Our method requires 2 steps: (a) Training models on polarly sentimented data and (b) computing the normalized loss for inputs for the polar models. The computed losses can then be modelled to generate the radii and theta for any inputs. You can follow these steps to do the same: 
### Train positive & negative models
To train the polar models (distilgpt) variant 
```
envs/bin/python3 train.py
```
To train the polar models (llama) variant (for weight interpolation) - If SLURM is available (sbatch) else omit the sbatch prefix from train.sh. Also edit num_processes depending on gpu availability. We use 4 A6000s. 
```
sbatch train.sh 
```
You may have to change the following parameters - num_processes (depending on how many gpus you have), output_dir (save path for your model) as well as wandb entity and run name as per your choice. You can also use ./sentiment_arithmetic/data_generator.py to create a new dataset compatible with the training format of the llama model (with preformatting). Internally, this file calls causal_ft.py for performing causal finetuning over the llama models.  
### Inference on Vanilla Dataset
```
envs/bin/python3 infer.py
```
### Inference on Augmented Dataset 
```
envs/bin/python3 infer_augment.py
```
You can test the correlation between Google API Scores and our polar labels as well. The preloaded values are the sentiment scores for the entire Yelp Test Split. 
```
python polar_api_correlation.py
```
## Classification baselines
### Multiclass
envs/bin/python3 00_baseline_sequence_classification.py
### Binary
envs/bin/python3 00_baseline_sequence_classification.py --num_labels 2


## Weight Vector Interpolation for Polar Model Probing
./sentiment_arithmetic/interpolate_models.py can be used to test different variants of the interpolated weights. You can specify range of alphas to test on within the file. You just need to specify the --base_model (pretrained Llama7B in this case), --pos_finetuned_model and --neg_finetuned_model as generated by your polar model training. 
```
python interpolate_models.py
```