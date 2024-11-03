conda create -p ./envs python=3.11
conda activate ./envs

conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
pip install pandas transformers[torch] datasets tensorboard matplotlib jupyterlab
