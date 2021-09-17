from gin import Net
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.distributions import Categorical
from ortools_tsp import solve


class Agentembedding(nn.Module):
    def __init__(self, node_feature_size, key_size, value_size):
        super(Agentembedding, self).__init__()
        self.key_size = key_size
        self.q_agent = nn.Linear(2 * node_feature_size, key_size)
        self.k_agent = nn.Linear(node_feature_size, key_size)
        self.v_agent = nn.Linear(node_feature_size, value_size)

    def forward(self, f_c, f):
        q = self.q_agent(f_c)
        k = self.k_agent(f)
        v = self.v_agent(f)
        u = torch.matmul(k, q.transpose(-1, -2)) / math.sqrt(self.key_size)
        u_ = F.softmax(u, dim=-2).transpose(-1, -2)
        agent_embedding = torch.matmul(u_, v)

        return agent_embedding


class AgentAndNode_embedding(torch.nn.Module):
    def __init__(self, in_chnl, hid_chnl, n_agent, key_size, value_size, dev):
        super(AgentAndNode_embedding, self).__init__()

        self.n_agent = n_agent

        # gin
        self.gin = Net(in_chnl=in_chnl, hid_chnl=hid_chnl).to(dev)
        # agent attention embed
        self.agents = torch.nn.ModuleList()
        for i in range(n_agent):
            self.agents.append(Agentembedding(node_feature_size=hid_chnl, key_size=key_size, value_size=value_size).to(dev))

    def forward(self, batch_graphs, n_nodes, n_batch):

        # get node embedding using gin
        nodes_h, g_h = self.gin(x=batch_graphs.x, edge_index=batch_graphs.edge_index, batch=batch_graphs.batch)
        nodes_h = nodes_h.reshape(n_batch, n_nodes, -1)
        g_h = g_h.reshape(n_batch, 1, -1)

        depot_cat_g = torch.cat((g_h, nodes_h[:, 0, :].unsqueeze(1)), dim=-1)
        # output nodes embedding should not include depot, refer to paper: https://www.sciencedirect.com/science/article/abs/pii/S0950705120304445
        nodes_h_no_depot = nodes_h[:, 1:, :]

        # get agent embedding
        agents_embedding = []
        for i in range(self.n_agent):
            agents_embedding.append(self.agents[i](depot_cat_g, nodes_h_no_depot))

        agent_embeddings = torch.cat(agents_embedding, dim=1)

        return agent_embeddings, nodes_h_no_depot


class Policy(nn.Module):
    def __init__(self, in_chnl, hid_chnl, n_agent, key_size_embd, key_size_policy, val_size, clipping, dev):
        super(Policy, self).__init__()
        self.c = clipping
        self.key_size_policy = key_size_policy
        self.key_policy = nn.Linear(hid_chnl, self.key_size_policy).to(dev)
        self.q_policy = nn.Linear(val_size, self.key_size_policy).to(dev)

        # embed network
        self.embed = AgentAndNode_embedding(in_chnl=in_chnl, hid_chnl=hid_chnl, n_agent=n_agent,
                                            key_size=key_size_embd, value_size=val_size, dev=dev)

    def forward(self, batch_graph, n_nodes, n_batch):

        agent_embeddings, nodes_h_no_depot = self.embed(batch_graph, n_nodes, n_batch)

        k_policy = self.key_policy(nodes_h_no_depot)
        q_policy = self.q_policy(agent_embeddings)
        u_policy = torch.matmul(q_policy, k_policy.transpose(-1, -2)) / math.sqrt(self.key_size_policy)
        imp = self.c * torch.tanh(u_policy)
        prob = F.softmax(imp, dim=-2)

        return prob


def action_sample(pi):
    dist = Categorical(pi.transpose(2, 1))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob


def get_reward(action, data, n_agent):

    subtour_max_lengths = [0 for _ in range(data.shape[0])]
    data = data * 1000
    depot = data[:, 0, :].tolist()
    sub_tours = [[[] for _ in range(n_agent)] for _ in range(data.shape[0])]
    for i in range(data.shape[0]):
        for n, m in zip(action.tolist()[i], data.tolist()[i][1:]):
            sub_tours[i][n].append(m)
        for tour in sub_tours[i]:
            tour.append(depot[i])

    for k in range(data.shape[0]):
        for a in range(n_agent):
            instance = sub_tours[k][a]
            sub_tour_length = solve(instance)/1000
            if sub_tour_length >= subtour_max_lengths[k]:
                subtour_max_lengths[k] = sub_tour_length
    return subtour_max_lengths


if __name__ == '__main__':
    from torch_geometric.data import Data
    from torch_geometric.data import Batch

    dev = 'cpu'
    torch.manual_seed(2)

    n_agent = 4
    n_nodes = 6
    n_batch = 3
    # get batch graphs data list
    fea = torch.randint(low=0, high=100, size=[n_batch, n_nodes, 2]).to(torch.float)  # [batch, nodes, fea]
    adj = torch.ones([fea.shape[0], fea.shape[1], fea.shape[1]])
    data_list = [Data(x=fea[i], edge_index=torch.nonzero(adj[i]).t()) for i in range(fea.shape[0])]
    # generate batch graph
    batch_graph = Batch.from_data_list(data_list=data_list).to(dev)

    # test policy
    policy = Policy(in_chnl=fea.shape[-1], hid_chnl=32, n_agent=n_agent, key_size_embd=64,
                    key_size_policy=64, val_size=64, clipping=10, dev=dev)

    pi = policy(batch_graph, n_nodes, n_batch)

    grad = torch.autograd.grad(pi.sum(), [param for param in policy.parameters()])

    action, log_prob = action_sample(pi)
    # print(log_prob)

    rewards = get_reward(action, fea, n_agent)

    print(rewards)


    # print(fea)
    # sub_tours = [[fea[b, 0]] for b in range(fea.shape[0])]
    # fea_repeat = fea.repeat(1, n_agent, 1).reshape(n_batch, n_agent, n_nodes, -1)
    # print(fea_repeat)
    # action_repeat = action + torch.arange(0, n_agent, n_agent*n_batch)
    # print(torch.arange(0, n_agent, n_agent*n_batch))
    # index_ops = (torch.arange(0, 3, 6)[:, None] + torch.arange(3)).view(-1)
    # print(index_ops)



    # grad1 = torch.autograd.grad(pi.sum(), [param for param in embd_net.parameters()])
    # print(grad1)
    # grad2 = torch.autograd.grad(pi.sum(), [param for param in policy.parameters()])
    # print(grad2)