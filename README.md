# Word-level-adversarial-defense
Deep Learning Project HS21

## Dependencies
- torch
- torchtext
- tqdm
- textattack
- scikit-learn

## BERT resources

- https://github.com/CSCfi/machine-learning-scripts/blob/master/examples/pytorch-imdb-bert.py
- https://www.kaggle.com/atulanandjha/bert-testing-on-imdb-dataset-extensive-tutorial
- https://github.com/hanxiao/bert-as-service

## Running on Euler
- Load the glove vectors to your scratch space on Euler (preferably using `scp`); it should mimic the way pytorch does it, i.e. just copy the zip from the `.vector_cache` that pytorch creates and unzip in the scratch on Euler. Don't forget to set the path variable `VECTOR_CACHE` in main to point to the glove folder in the scratch.
- Load the datasets to your scratch space on Euler (preferably using `scp`); similar to the first point, do so with the zips that pytorch downloads into a cache folder. Unpack in Euler scratch and set path variable in `datasets_euler.py`.
- (Has to be done everytime you log on to Euler) Switch to new software stack (`env2lmod`), load gcc 8.2.0 and python 3.8.5 for GPUs (`module load gcc/8.2.0 python_gpu/3.8.5`).
- Create a Python environment using `python -m venv --system-site-packages my_venv` and activate using `source $HOME/my_venv/bin/activate`. Install textattack and torchtext, make sure to install torchtext 0.10.1 to not load a newer torch version `pip install torchtext=0.10.1 textattack`. If any other packages miss, also install through pip.
- Job submission works as usual, e.g. `bsub -n 4 -We 05:00 -R "rusage[mem=3000,ngpus_excl_p=1]" -o output.txt python main.py`. This creates a job for four cores, each getting 3000MB RAM using one GPU, expected runtime is 5h and the output is written to output.txt.
- Upload the transformers cache also to your scratch space. On Euler, set the environment variable `TRANSFORMERS_CACHE` to the folder you uploaded the transformers cache to. Also set `TRANSFORMERS_OFFLINE=1`.
