# GNN-Motion-Planning

[[project page](https://rainorangelemon.github.io/NeurIPS2021/)] [[paper](https://rainorangelemon.github.io/NeurIPS2021/paper.pdf)]

![framework](./data/images/framework.pdf)
![envs](./data/images/envs.png)

## Abstract

Sampling-based motion planning is a popular approach in robotics for finding paths in continuous configuration spaces. Checking collision with obstacles is the major computational bottleneck in this process. We propose new learning-based methods for reducing collision checking to accelerate motion planning by training graph neural networks (GNNs) that perform path exploration and path smoothing. Given random geometric graphs (RGGs) generated from batch sampling, the path exploration component iteratively predicts collision-free edges to prioritize their exploration. The path smoothing component then optimizes paths obtained from the exploration stage. The methods benefit from the ability of GNNs of capturing geometric patterns from RGGs through batch sampling and generalize better to unseen environments. Experimental results show that the learned components can significantly reduce collision checking and improve overall planning efficiency in challenging high-dimensional motion planning tasks.

## Installation
```bash
conda create -n gnnmp python=3.8
conda activate gnnmp
# install pytorch, modify the following line according to your environment
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# install torch geometric, refer to https://github.com/pyg-team/pytorch_geometric
conda install pyg -c pyg
pip install pybullet jupyterlab transforms3d matplotlib shapely descartes
```

## Download Extra Files for Training / Test
Please find the training and test sets in https://drive.google.com/drive/folders/1ADHjbmhT7OCFmOSLn4YDZF0CBmkmzLye?usp=sharing. Once the files are downloaded, put them under the `data/pkl` folder.

## Usage
See [main.ipynb](./main.ipynb) for further details.

To evaluate all the methods, see [eval_all.py](./eval_all.py)
