import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import trange
import torch

from model import Model
from agent import Agent
from system import System
from randomStateDataset import RandomStateDataset

if __name__ == '__main__':
    
    num_epoch = 300
    batch_size = 1000
    cur_length = 2
    full_dataset_length = 5 # generated data list length
    max_length = 50
    update_interval = 100
    num_samples = batch_size * update_interval
    loss_tolerance = 0.01
    accuracy_tolerance = 0.001 # stop searching when distance less than accuracy_tolerance
    result_dir = 'mytry/results/'
    ckpt_dir = 'mytry/ckpts/'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = System(device)
    
    policy_net = Model(env, input_size=8, embedding_size=5000, hidden_size=1000).to(device)
    target_net = Model(env, input_size=8, embedding_size=5000, hidden_size=1000).to(device)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
        
    agent = Agent(policy_net, target_net, env, accuracy_tolerance)
    dataset = RandomStateDataset(env, cur_length, full_dataset_length, max_length, num_samples, accuracy_tolerance)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)

    while cur_length < max_length:   
        is_updated = 0
        for n_epoch in trange(num_epoch):
            dataset.reinitialize()
            ave_loss = 0
            for sample in dataloader:
                loss = agent.update_model(sample)
                ave_loss += loss
            ave_loss /= update_interval
            print('loss:', ave_loss, 'cur_len:', cur_length, flush=True)
            if n_epoch % 10 == 0:
                if ave_loss < loss_tolerance:
                    target_net.load_state_dict(policy_net.state_dict())
                    is_updated = 1
        if is_updated:
            cur_length += 1
            dataset.cur_length += 1
            loss_tolerance = 0.01
        else:
            loss_tolerance += 0.001
        num_epoch += 10
        torch.save(policy_net.state_dict(), ckpt_dir+'model_{}_{}.ckpt'.format(num_epoch, cur_length)) 
