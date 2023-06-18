# DP-SGD with random sparsification

A demonstration for applying RS on [DP-CNN](https://ojs.aaai.org/index.php/AAAI/article/view/17123)


## Installation
```bash
pip install -r requirements.txt
```

## Usage

DP-SGD with random sparsification.
```bash
python train.py --epochs 48 --lr 1 --batchsize 1000 --clip 1 --momentum 0 --eps 3 --final-rate 0.9
```

DP-SGD without random sparsification, but tuned based on our scaling rule.
```bash
python train.py --epochs 40 --lr 1 --batchsize 1000 --clip 1 --momentum 0 --eps 3 --final-rate 0
```

DP-SGD using hyperparameters given in previous work.
```bash
python train.py --epochs 40 --lr 1 --batchsize 1000 --clip 0.1 --momentum 0.9 --eps 3 --final-rate 0
```
