import os
from tqdm import trange
import numpy as np
import torch
torch.multiprocessing.set_start_method("spawn", force=True)

from model import Model
from agent import Agent
from system import System
# from testDataset import TestDataset

num_epoch = 630 # 1320
batch_size = 20000
min_length = 1
cur_length = 11 # 45
full_dataset_length = 5
max_length = cur_length
num_samples = batch_size
accuracy_tolerance = 0.001

ckpt_dir = 'mytry/ckpts/'
result_dir = 'mytry/results/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = System(device)
policy_net = Model(env, input_size=8, embedding_size=5000, hidden_size=1000).to(device)
target_net = Model(env, input_size=8, embedding_size=5000, hidden_size=1000).to(device)

policy_net.load_state_dict(torch.load(ckpt_dir + 'model_{}_{}.ckpt'.format(num_epoch, cur_length), map_location=device))

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

agent = Agent(policy_net, target_net, env, accuracy_tolerance)

brute_force_length = 9
maximum_depth = 100
expand_size = 10000 # 3000 highest-scoring samples
keep_size = 100000
n_sample = 20 # 50

SEED = 19
torch.manual_seed(SEED)
targets = env.randSU(n_sample)

# targets = env.unitary2(1)

min_dists = []
seq_lengths = []

for i in trange(len(targets)):
    state = targets[i]
    min_dist, best_state, best_seq = agent.search(state, brute_force_length, expand_size, keep_size, maximum_depth)
    state_np = best_state[0].detach().cpu().numpy() + 1j * best_state[1].detach().cpu().numpy()
    state_np /= (np.linalg.det(state_np)) ** (1.0/2.0) 
    min_dists.append(min_dist.detach().cpu().numpy().item())
    seq_lengths.append(torch.sum((best_seq != -1).float()).detach().cpu().numpy().item())
    
    print('min_dist:', min_dist)
    print('best_state:', state_np)
    print('best_seq:', best_seq)
print('average distance:', sum(min_dists)/len(min_dists))
print('average length:', sum(seq_lengths)/len(min_dists))





