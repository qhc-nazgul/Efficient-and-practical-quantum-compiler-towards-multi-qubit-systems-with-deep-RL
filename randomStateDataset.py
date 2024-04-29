import numpy as np
import torch
from torch.utils import data
from dataGenerator import DataGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RandomStateDataset(data.Dataset):
    def __init__(self, env, cur_length, full_dataset_length, max_length, num_samples, epsilon):
        self.env = env
        self.cur_length = cur_length
        self.max_length = max_length
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.full_dataset_length = full_dataset_length

        self.generator = DataGenerator(env, epsilon)
        self.states_full, self.actions_list_full, self.actions, self.next_states_full,\
            self.masks_full = self.generator.calc_data_full(self.full_dataset_length)       
        n = self.cur_length - self.full_dataset_length
        if n > 0:
            self.states_rand, self.actions_list_rand, self.next_states_rand, self.masks_rand\
                = self.generator.calc_data_rand(self.states_full[-1].view(-1, 2, 2, 2), self.actions, n)
            
    def reinitialize(self):
        self.states_full, self.actions_list_full, self.actions, self.next_states_full,\
            self.masks_full = self.generator.calc_data_full(self.full_dataset_length)    
        n = self.cur_length - self.full_dataset_length
        if n > 0:
            self.states_rand, self.actions_list_rand, self.next_states_rand, self.masks_rand\
                = self.generator.calc_data_rand(self.states_full[-1].view(-1, 2, 2, 2), self.actions, n)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, _):
        length = torch.randint(0, self.cur_length, ())
        if length < self.full_dataset_length:
            idx = torch.randint(0, len(self.states_full[length]), ())
            state = self.states_full[length][idx]
            actions = self.actions_list_full[length][idx]
            next_states = self.next_states_full[length][idx]
            mask = self.masks_full[length][idx]
        else:
            length = length - self.full_dataset_length
            idx = torch.randint(0, len(self.states_rand[length]), ())
            state = self.states_rand[length][idx]
            actions = self.actions_list_rand[length][idx]
            next_states = self.next_states_rand[length][idx]
            mask = self.masks_rand[length][idx]            
        return {'state': state, 'actions': actions, 'next_states': next_states, 'mask': mask}