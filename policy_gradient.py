# Adapted from https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
import sys
import torch  
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, action_dim, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()

        self.action_dim = action_dim
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, action_dim)

        # Log std as a learnable parameter (optional: you can also output it from a layer)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        # print("LINEAR1: ", self.linear1)
        print("Weight:", self.linear1.weight) 
        print("Bias:", self.linear1.bias) 
        # print("Linear:", self.linear1(state))
        # print("STATE:", state)
        # print("X:", x) #TODO: fix situation where X = [nan, nan, nan, ...]. state has reasonable values, so problem must be elsewhere
        # weights and bias have nan, so that may be where the issue originates
        mean = self.mean_layer(x)
        std = self.log_std.exp().expand_as(mean)  # convert log_std to std
        return mean, std

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        mean, std = self.forward(Variable(state))
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # sum over action dims
        return action.squeeze(0).detach().numpy(), log_prob