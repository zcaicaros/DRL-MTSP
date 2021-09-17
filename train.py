from policy import Policy, action_sample, get_reward
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from validation import validate


def train(no_batch, no_nodes, policy_net, l_r, no_agent, iterations, device):

    # prepare validation data
    validation_data = torch.load('./validation_data_'+str(no_nodes)+'_'+str(1000))
    # a large start point
    best_so_far = 1000000
    validation_results = []

    # optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=l_r)

    for itr in range(iterations):
        # prepare training data
        data = torch.rand(size=[no_batch, no_nodes, 2])  # [batch, nodes, fea], fea is 2D location
        adj = torch.ones([data.shape[0], data.shape[1], data.shape[1]])  # adjacent matrix fully connected
        data_list = [Data(x=data[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t()) for i in range(data.shape[0])]
        batch_graph = Batch.from_data_list(data_list=data_list).to(device)

        # get pi
        pi = policy_net(batch_graph, n_nodes=data.shape[1], n_batch=no_batch)
        # sample action and calculate log probabilities
        action, log_prob = action_sample(pi)
        # get reward for each batch
        reward = get_reward(action, data, no_agent)  # reward: tensor [batch, 1]
        # compute loss
        loss = torch.mul(torch.tensor(reward, device=device) - 2, log_prob.sum(dim=1)).sum()

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if itr % 100 == 0:
            print('\nIteration:', itr)
        print(format(sum(reward)/no_batch, '.4f'))

        # validate and save best nets
        if (itr+1) % 100 == 0:
            validation_result = validate(validation_data, policy_net, no_agent, device)
            if validation_result < best_so_far:
                torch.save(policy_net.state_dict(), './{}.pth'.format(
                    str(no_nodes) + '_' + str(no_agent)))
                print('Found better policy, and the validation result is:', format(validation_result, '.4f'))
                validation_results.append(validation_result)
                best_so_far = validation_result
    return validation_results


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(2)

    n_agent = 5
    n_nodes = 50
    batch_size = 512
    lr = 1e-4
    iteration = 20000

    policy = Policy(in_chnl=2, hid_chnl=32, n_agent=n_agent, key_size_embd=64,
                    key_size_policy=64, val_size=64, clipping=10, dev=dev)

    best_results = train(batch_size, n_nodes, policy, lr, n_agent, iteration, dev)
    print(min(best_results))
