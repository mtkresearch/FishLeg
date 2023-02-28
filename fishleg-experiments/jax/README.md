# FishLeg Jax Implementation

## Create and activate a new `python` environment:

### Using `virtualenv`
This assumes `python3` is installed and it's at version 3.8.10
```
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
python3 -m venv ~/.virtualenvs/fishleg
source ~/.virtualenvs/fishleg/bin/activate
```

### Using `conda`
```
conda create -n fishleg python=3.8
conda activate fishleg
```

## Install the requiremed packages
```
pip install -r requirements.txt
```

## (Optional but recommended) Install Jax with GPU support
```
pip install --upgrade "jax[cuda]==0.3.14" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Run the script
```
python3 main.py -c configs/config_example.json
```