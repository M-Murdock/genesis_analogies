# Loads an RL policy

import numpy as np
import torch
from policy_gradient import PolicyNetwork
import matplotlib.pyplot as plt
from kitchen_gym import KitchenGym

## Default Args
DEFAULT_RADIUS = 0.034
DEFAULT_HEIGHT = 0.09
DEFAULT_RHO = 2000
DEFAULT_FRICTION = 0.5
DEFAULT_STARTING_X = 0.65 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Genesis Gym Environment')
    parser.add_argument('--vis', action='store_true', help='Enable visualization')
    parser.add_argument('--radius', type=float, default=DEFAULT_RADIUS, help='Bottle radius')
    parser.add_argument('-e', '--height', type=float, default=DEFAULT_HEIGHT, help='Bottle height')
    parser.add_argument('-o', '--rho', type=float, default=DEFAULT_RHO, help='Density of the bottle')
    parser.add_argument('--friction', type=float, default=DEFAULT_FRICTION, help='Friction of the bottle')
    parser.add_argument('--starting_x', type=float, default=DEFAULT_STARTING_X, help='Starting x position of the bottle')
    parser.add_argument('--max-demos', type=int, default=1e7, help='Max number of demos to load')
    args = parser.parse_args()
    

    env = KitchenGym(args)

    PATH="kitchen_model.pt"
    policy_net = torch.load(PATH, weights_only=False)
    

    if args.vis: env.render(use_imshow=True)
    
    max_episode_num = 5000
    max_steps = 1000

    PATH="kitchen_model.pt"
    for episode in range(max_episode_num):

        print("Episode ", episode)
        # reset the state
        state = env.reset()["state"]

        # walk through each step
        for steps in range(max_steps):
            print("Step ", steps)

            # get the next action 
            action, log_prob = policy_net.get_action(state)

            # get the resulting state and reward
            new_state, _, done, _ = env.step(action)
            
            if done:
                print("Done!")
                break

            state = new_state["state"]
            
