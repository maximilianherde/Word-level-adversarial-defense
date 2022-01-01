# Word-level Adversarial Defense Layer for Robust Natural Language Classification
Project of the course __Deep Learning__ (HS21) @ ETH Zurich.

## Code structure
The main files are given by:
- **main.py:** Models can be trained using this script.
- **attack.py:** This script is for conducting attacks on the previously trained models.
- **adv_dataset_generator.py:** Do vanilla adversarial training using this script (BiLSTM, PWWS).
- **sem.py:** Create SEM embeddings with this script.

The implementation of the models can be found in the `models` subfolder, functions for the calculation of metrics can be found in the `metrics` subfolder. `attackutils/modelwrapper.py` contains all custom model wrappers needed for attacking the models. `WLADL.py` accommodates our defense layer.

Classes for the datasets can be found in `datasets_euler.py`. `datasets.py` is a deprecated version for the three classification datasets. In `datasets_wrapped_ta.py` are getter functions for datasets that can be used with TextAttack.

The `notebooks` subdirectory contains Jupyter Notebooks that were used during the project since some of the work was done in Google Colab.

## Dependencies
- torch
- torchtext
- torchfile
- transformers
- tqdm
- textattack
- scikit-learn

## Reproducing results
Results can be reproduced by training the models using `main.py` and then attacking using `attack.py`.

## Running on Euler, ETHZ's computing cluster
For a lot of our experiments we used GPUs from the Euler cluster. To run our code on the cluster, we use GCC 8.2.0 as Python compiler backend and Python 3.8.5 (switch to new software stack using `env2lmod`, load modules `module load gcc/8.2.0 python_gpu/3.8.5`). Since some packages we rely on are missing, installing them is necessary. To do so, create a virtual Python environment in the home directory (`python -m venv --system-site-packages <NAME_OF_VENV>`) and activate it (`source $HOME/<NAME_OF_VENV>/bin/activate`). Then install torchfile, textattack, torchtext==0.10.1 and tensorflow-text using pip (`pip install torchfile textattack torchtext==0.10.1 tensorflow-text`). If for any reason this list is not complete, also install using pip. Since Euler compute nodes are a firewalled environment, all data that would in an easy setting be downloaded by PyTorch or TextAttack must be uploaded manually onto Euler (to the scratch space), i.e. upload the GloVe embeddings used and the datasets. Make sure to comply with the structure that is expected by the files. For the transformers model data and the TextAttack data, do a dry run using the attacks and models on your own machine until actual training/attacking starts. Then, upload the cached data to Euler.

Before running anything, make sure to set two/three environment variables:
- `export TRANSFORMERS_OFFLINE=1`, always
- `export TRANSFORMERS_CACHE=<PATH_TO_TRANSFORMERS_CACHE>`, always
- `export TA_CACHE_DIR=<PATH_TO_TEXTATTACK_CACHE>`, when attacking
