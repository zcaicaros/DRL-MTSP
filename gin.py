import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_mean_pool


class Net(torch.nn.Module):
    def __init__(self, in_chnl, hid_chnl):
        super(Net, self).__init__()

        ## init projection
        # 1st mlp layer
        self.lin1_1 = torch.nn.Linear(in_chnl, hid_chnl)
        self.bn1_1 = torch.nn.BatchNorm1d(hid_chnl)
        self.lin1_2 = torch.nn.Linear(hid_chnl, hid_chnl)

        ## GIN conv layers
        nn1 = Sequential(Linear(hid_chnl, hid_chnl), ReLU(), Linear(hid_chnl, hid_chnl))
        self.conv1 = GINConv(nn1, eps=0, train_eps=False, aggr='mean')
        self.bn1 = torch.nn.BatchNorm1d(hid_chnl)
        nn2 = Sequential(Linear(hid_chnl, hid_chnl), ReLU(), Linear(hid_chnl, hid_chnl))
        self.conv2 = GINConv(nn2, eps=0, train_eps=False, aggr='mean')
        self.bn2 = torch.nn.BatchNorm1d(hid_chnl)
        nn3 = Sequential(Linear(hid_chnl, hid_chnl), ReLU(), Linear(hid_chnl, hid_chnl))
        self.conv3 = GINConv(nn3, eps=0, train_eps=False, aggr='mean')
        self.bn3 = torch.nn.BatchNorm1d(hid_chnl)
        # nn4 = Sequential(Linear(hid_chnl, hid_chnl), ReLU(), Linear(hid_chnl, hid_chnl))
        # self.conv4 = GINConv(nn4, eps=0, train_eps=False, aggr='mean')
        # self.bn4 = torch.nn.BatchNorm1d(hid_chnl)

        ## layers used in graph pooling
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(1+3):  # 1+x: 1 projection layer + x GIN layers
            self.linears_prediction.append(nn.Linear(hid_chnl, hid_chnl))

    def forward(self, x, edge_index, batch):

        # init projection
        h = self.lin1_2(F.relu(self.bn1_1(self.lin1_1(x))))
        hidden_rep = [h]

        # GIN conv
        h = F.relu(self.bn1(self.conv1(h, edge_index)))
        node_pool_over_layer = h
        hidden_rep.append(h)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        node_pool_over_layer += h
        hidden_rep.append(h)
        h = F.relu(self.bn3(self.conv3(h, edge_index)))
        node_pool_over_layer += h
        hidden_rep.append(h)
        # h = F.relu(self.bn4(self.conv4(h, edge_index)))
        # node_pool_over_layer += h
        # hidden_rep.append(h)

        gPool_over_layer = 0
        # Graph pool
        for layer, layer_h in enumerate(hidden_rep):
            g_pool = global_mean_pool(layer_h, batch)
            gPool_over_layer += F.dropout(self.linears_prediction[layer](g_pool),
                                          0.5,
                                          training=self.training)

        return node_pool_over_layer, gPool_over_layer

