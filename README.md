# A reinforcement learning approach for optimizing multiple traveling salesman problems over graphs


## Some differences compared with the original paper

 - We use GIN as the graph embedding network
 - We use a shared self-attention module for customer assignment 

## Installation
python 3.8.8

CUDA 11.1

pytorch 1.9.0 with CUDA 11.1

[PyG](https://github.com/pyg-team/pytorch_geometric) 1.7.2


[Or-Tools](https://github.com/google/or-tools) 9.0.9048

Then install dependencies:
```
pip install --upgrade pip
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-geometric==1.7.2
pip install ortools==9.0.9048
```

## Training
```
# generate validation dataset
python3 data_generator.py
python3 train.py
```
