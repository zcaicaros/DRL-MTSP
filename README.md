# A reinforcement learning approach for optimizing multiple traveling salesman problems over graphs


## Some differences compared with the original paper

 - We use GIN as the graph embedding network.
 - We use a shared self-attention module for customer assignment.
 - The official implementation can be found [Here](https://github.com/YujiaoHu/MinMax-MTSP).
 - Paper reference:

```
@article{hu2020reinforcement,
  title={A reinforcement learning approach for optimizing multiple traveling salesman problems over graphs},
  author={Hu, Yujiao and Yao, Yuan and Lee, Wee Sun},
  journal={Knowledge-Based Systems},
  volume={204},
  pages={106244},
  year={2020},
  publisher={Elsevier}
}
```
 - We claim that our implementation has the better performance than the original one.
## Installation
python 3.8.8

CUDA 11.1

pytorch 1.9.0 with CUDA 11.1

[PyG](https://github.com/pyg-team/pytorch_geometric) 1.7.2


[Or-Tools](https://github.com/google/or-tools) 9.0.9048

Then install dependencies:
```
cd ~
pip install --upgrade pip
pip install torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install torch-geometric==1.7.2
pip install ortools==9.0.9048
```

## Use code
### Training
```
python3 train.py
```
### Testing
```
python3 test.py
```
### Generate validation and testing dataset
Adjust parameters in `data_generator.py`, then run `python3 data_generator.py`
