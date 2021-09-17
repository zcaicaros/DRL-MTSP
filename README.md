# A reinforcement learning approach for optimizing multiple traveling salesman problems over graphs

## Installation
python 3.8.8

CUDA 11.1

pytorch 1.9.0 with CUDA 11.1

[PyG](https://github.com/pyg-team/pytorch_geometric) 1.7.2


[Or-Tools](https://github.com/google/or-tools) 9.0.9048

Then install dependencies:
```
pip install --user --upgrade pip
pip install --user torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install --user torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install --user torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install --user torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install --user torch-geometric==1.7.2
pip install ortools==9.0.9048
```

## Training
```
python3 train.py
```
