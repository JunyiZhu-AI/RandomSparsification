## Improving Differentially Private SGD via Randomly Sparsified Gradients [Accepted by TMLR]

In this codebase, we showcase the enhancement of DP-SGD through the random sparsification of gradients, employing a simple example with [DP-CNN](https://ojs.aaai.org/index.php/AAAI/article/view/17123). The interplay between DP-SGD and Random Sparsification (RS) is peculiar and indicative, as it diverges from observations made in other popular SGD schemes.

### Abstract
Differentially private stochastic gradient descent (DP-SGD) has been widely adopted in deep learning to provide rigorously defined privacy, which requires gradient clipping to bound the maximum norm of individual gradients and additive isotropic Gaussian noise. With analysis of the convergence rate of DP-SGD in a non-convex setting, we identify that randomly sparsifying gradients before clipping and noisification adjusts a trade-off between internal components of the convergence bound and leads to a smaller upper bound when the noise is dominant. Additionally, our theoretical analysis and empirical evaluations show that the trade-off is not trivial but possibly a unique property of DP-SGD, as either canceling noisification or gradient clipping eliminates the trade-off in the bound. This observation is indicative, as it implies DP-SGD has special inherent room for (even simply random) gradient compression. To verify the observation an utilize it, we propose an efficient and lightweight extension using random sparsification (RS) to strengthen DP-SGD.  Experiments with various DP-SGD frameworks show that RS can improve performance. Additionally, the produced sparse gradients of RS exhibit advantages in reducing communication cost and strengthening privacy against reconstruction attacks, which are also key problems in private machine learning.

### Installation
Make sure that conda is installed.
```sh
git clone git@github.com:JunyiZhu-AI/RandomSparsification.git
cd RandomSparsification
conda create -n rs python==3.9.12
conda activate rs
conda install pip
pip install -r requirement.txt
```

### Usage

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

In the paper, we consistently perform per-epoch randomization in consideration of the communicational efficiency. However, randomizing the sparsification mask multiple times per epoch may further improve the performance of the network.
```bash
python train.py --epochs 40 --lr 1 --batchsize 1000 --clip 0.1 --momentum 0.9 --eps 3 --final-rate 0.9 --refresh 5
```
### Citation
```sh
@misc{zhu2022improving,
      title={Improving Differentially Private SGD via Randomly Sparsified Gradients}, 
      author={Junyi Zhu and Matthew B. Blaschko},
      year={2022},
      eprint={2112.00845},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
