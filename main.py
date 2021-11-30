import torch
import numpy as np
from environment import MazeEnv
from environment.graph import Graph
from next_model import GNN
from agent import Agent
from replay import Replay
from config import config, set_random_seed
from tqdm import tqdm as tqdm
from tensorboardX import SummaryWriter
from time import time

dones = 0.
successes = 0.

writer = SummaryWriter()

set_random_seed(1234)
epoch = 2000; step_max = 100; iter = 1;
env = MazeEnv(dim=2)
model = GNN(gnn_in=8, gnn_out=3, gnn_layers=3, embed_dim=2, iter=10, lstm_size=5)
agent = Agent(model, Replay(), **config)

T = 0
for index in tqdm(range(epoch)):
    env.init_new_problem(index)
    set_random_seed(index)
    env.set_random_init_goal()

    graph = Graph(env).initialize(500, self_loop=True)
    start_time = time()

    if np.sum(graph.E_mask)==0:
        dones += 1
        continue

    j = 0
    while j < step_max and (time()-start_time<60):
        steps, done, success = agent.act(graph, env)
        j += steps
        if done:
            break

    agent.buffer(env, graph)
    loss_weights = agent.learn(iter)
    T += 1
    if loss_weights is not None:
        loss, weights = loss_weights
        writer.add_scalar('total_loss', loss, T)
        writer.add_scalar('weights', weights, T)

    torch.save(agent.model.state_dict(), 'weights.pt')

    dones += done
    successes += success
    writer.add_scalar('done', dones/float(index+1), T)
    writer.add_scalar('success', successes/float(index+1), T)

torch.save(agent.model.state_dict(), 'weights.pt')
writer.close()

print(dones, successes)