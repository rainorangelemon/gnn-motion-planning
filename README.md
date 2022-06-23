# GNN-Motion-Planning

[project page](https://rainorangelemon.github.io/NeurIPS2021/) [paper](https://rainorangelemon.github.io/NeurIPS2021/paper.pdf)

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

## Play
See [main.ipynb](./main.ipynb) for further details.
