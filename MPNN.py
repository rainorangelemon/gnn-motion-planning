# Define the MNN model that will be trained

# Part of the code is based on https://github.com/timlacroix/nri_practical_session

import dgl
from generate_dataset import DatasetGenerator
import torch
import torch.nn as nn

# Setting the seed for replicability
import random

random.seed(33)


# Do not use DGL in the end?

# A first NN for a single algorithm
# Define the MPNN module
# (Use of DFS for the first exemple)

class MPNN(nn.Module):
    # Expects dgl graphs as inputs
    def __init__(self, in_feats, hidden_feats, edge_feats, out_feats, useCuda=False):
        super(MPNN, self).__init__()
        self.n_hid = hidden_feats
        self.encoder = nn.Linear(in_feats + hidden_feats + 1,
                                 hidden_feats)  # +1 is for the weights (needed so far, might be removed later)
        self.M = nn.Linear(hidden_feats * 2 + edge_feats, 32)
        self.U = nn.Linear(hidden_feats * 2, hidden_feats)
        # self.decoder = Linear_layer(hidden_feats * 2 , in_feats) # "first" version, does not account for next node prediction
        self.decoder_nextnode = nn.Linear(hidden_feats * 2,
                                          1)  # output "energy" will be soft-maxed to predict next node
        self.decoder_update = nn.Linear(hidden_feats * 2 + 1,
                                        in_feats)  # takes the same inputs + next_node "energy" and computes updates
        self.termination = nn.Linear(hidden_feats, 1)  # Find a way to have only 1 outputs whatever the graph size is
        self.useCuda = useCuda

    def compute_send_messages(self, edges):
        # The argument is a batch of edges.
        # This computes a (batch of) message called 'msg' using the source node's feature 'h'.
        z_src = edges.src['z']
        z_dst = edges.dst['z']

        msg = self.M(torch.cat([z_src, z_dst, edges.data['features'].view(-1, 1)], 1))
        return {'msg': msg}

    def max_reduce_messages(self, nodes):
        # The argument is a batch of nodes.
        # This computes the new 'h' features by summing received 'msg' in each node's mailbox.
        # return {'u_input' : torch.sum(nodes.mailbox['msg'], dim=1)} # for sum and mean: add ''
        return {'u_input': torch.max(nodes.mailbox['msg'], dim=1).values}

    # A step corresponds to 1 iteration of the network:
    # Giving the state of the graph after one iteration of the algrithm
    def step(self, graph, inputs, hidden):

        # Helpers to stack conviniently z and e
        n_atoms = inputs.size(0)
        id1 = torch.LongTensor(sum([[i] * n_atoms for i in range(n_atoms)], []))
        id2 = torch.LongTensor(sum([list(range(n_atoms)) for i in range(n_atoms)], []))

        # Encoding x^t and h^{t-1}
        inputs = inputs.view(-1, 1)
        inp = torch.cat([inputs, hidden, graph.ndata['priority'].view(-1, 1)], 1)
        z = self.encoder(inp)
        graph.ndata['z'] = z

        # Processor
        # without dgl: messages = self.M(stack) but hard to pass on messages
        # Extract the aggregation(max) of messages at each position
        # Easier with DGL:
        graph.send(graph.edges(), self.compute_send_messages)
        # trigger aggregation at all nodes
        graph.recv(graph.nodes(), self.max_reduce_messages)

        u_input = graph.ndata.pop('u_input')

        new_hidden = self.U(torch.cat([z, u_input], 1))

        # Stoping criterion for the next step
        H_mean = torch.mean(new_hidden, dim=0, keepdim=True)

        # TODO: find right way to broadcast
        loc_inp = torch.cat([new_hidden, H_mean])
        loc_out = self.termination(loc_inp)
        m = nn.Sigmoid()
        stop = m(loc_out)
        stop = torch.max(stop).view((1, 1))

        # Decoder
        # new_state = self.decoder(torch.cat([new_hidden, z], 1)) # first version
        next_node_energy = self.decoder_nextnode(torch.cat([new_hidden, z], 1))
        # Add a message-passing between the two?
        new_state = self.decoder_update(torch.cat([new_hidden, z, next_node_energy], 1))

        return new_state, new_hidden, stop, next_node_energy

    # Iterate steps until completion
    def forward(self, graph, states, edges_mat):

        # Initialize hidden state at zero
        hidden = torch.zeros(states.size(1), self.n_hid).float()
        # print('Shape of hidden state:', hidden.size())

        # Store states and termination prediction
        pred_all = [states[0].view(-1, 1).float()]
        pred_stop = [torch.tensor([[0]]).float()]
        pred_nextnode = []

        # set all edges features inside graph (for easier message passing)
        edges_features = []
        for i in range(graph.edges()[0].size(0)):
            # Extract the features of each existing edge
            edges_features.append(edges_mat[graph.edges()[0][i], graph.edges()[1][i]])

        graph.edata['features'] = torch.FloatTensor(edges_features)

        if self.useCuda:
            graph.edata['features'] = graph.edata['features'].cuda()
            graph.ndata['priority'] = graph.ndata['priority'].cuda()
            hidden = hidden.cuda()
            pred_stop = [torch.tensor([[0]]).float().cuda()]

        # Iterate the algorithm for all steps
        for i in range(states.size(0) - 1):
            new_state, hidden, stop, next_node_energy = self.step(graph, pred_all[i], hidden)

            next_node_pred = next_node_energy  # nn.Softmax(dim=0)(next_node_energy) # Softmax is already done in CrossEntropyLoss from Pytorch?

            pred_all.append(new_state)
            pred_stop.append(stop)
            pred_nextnode.append(next_node_pred)

        preds = torch.stack(pred_all, dim=1).view(states.size(0), states.size(1))
        preds_stop = torch.stack(pred_stop, dim=1)
        preds_nextnode = torch.stack(pred_nextnode, dim=1)

        return preds, preds_stop, preds_nextnode