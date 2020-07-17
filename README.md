# WordCraft

### Setup
Create a conda environment and install dependencies:
```
conda env create -f environment.yml
conda activate commonsense-rl
pip install -r requirements.txt
```

Install spaCy model for GloVe embeddings:
```
python -m spacy download en_core_web_lg
```

### TextLab
Try running the environment in interactive mode:
```
python -m scripts.interactive
```

To make environments completely deterministic, you must first fix PYTHONHASHSEED in the shell by executing `export PYTHONHASHSEED="0"`.
You can turn random hashing back on by setting this value back to random: `export PYTHONHASHSEED="random"`.