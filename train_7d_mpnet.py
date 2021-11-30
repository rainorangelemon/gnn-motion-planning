import torch
import numpy as np
from environment import KukaEnv
from next_model import EncoderProcessDecoder
from torch_geometric.data import Data
from config import config, set_random_seed
from tqdm import tqdm as tqdm
from tensorboardX import SummaryWriter
import pickle
from time import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = None
INFINITY = float('inf')


env = KukaEnv()
set_random_seed(1234)
epoch = 2000; iter = 40; graph_size = 200; loop=30
model = EncoderProcessDecoder(workspace_size=3, config_size=7, embed_size=32, map_size=9261).to(device)
try:
    model.load_state_dict(torch.load('weights.pt', map_location=device))
except:
    pass
with open('kuka_prm.pkl', 'rb') as f:
    graphs = pickle.load(f)

T = 0
losses = []
model.train()
encoder_optimizer = torch.optim.Adam(set(model.parameters()) - set(model.next_node.parameters()), lr=config.lr)
decoder_optimizer = torch.optim.Adam(set(model.next_node.parameters()), lr=config.lr)
encoder_optimizer.zero_grad()
decoder_optimizer.zero_grad()
aggressive = True
pre_mi = 0
cur_mi = 0
loss = 0.

for iter_i in range(iter):
    indexes = np.random.permutation(epoch)
    pbar = tqdm(indexes)
    for index in pbar:

        index = index % 2000

        pb = env.init_new_problem(index)

        time0 = time()

        data = Data(x_obstacles=torch.FloatTensor(env.obstacles),
                    path=torch.FloatTensor(env.path))
        next_node_index = np.array([np.random.choice(len(data.path)-1)])
        goal_index = np.array([np.random.randint(next_node_index, len(data.path))])
        data.x_goal = data.path[goal_index]
        data.target = data.path[next_node_index]
        data.next_node = data.path[next_node_index+1]
        data.next_node = (data.next_node - data.target) / (1e-6 + (data.next_node - data.target).norm(dim=-1, keepdim=True))

        time_data = time() - time0

        time0 = time()
        next_node = model.reactive_policy(**vars(data.to(device)))
        next_node_loss = torch.nn.MSELoss()(next_node, data.next_node)
        loss = loss + next_node_loss
        losses.append(next_node_loss)
        time_train = time() - time0

        time0 = time()
        if T % 8 == 0:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            loss = 0.
            encoder_optimizer.step()
            decoder_optimizer.step()

            if writer is None:
                writer = SummaryWriter()
            writer.add_scalar('train/total_loss', sum(losses) / len(losses), T)
            losses = []

            torch.save(model.state_dict(), 'weights.pt')

        T += 1
        time_bp = time() - time0

        pbar.set_description("data %.2fs, train %.2fs, bp %.2fs" % (time_data, time_train, time_bp))

    torch.save(model.state_dict(), 'weights_%s.pt' % str(iter_i))

torch.save(model.state_dict(), 'weights.pt')
writer.close()
