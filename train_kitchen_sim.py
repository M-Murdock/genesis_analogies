# Trains a simple RL policy
# Implements REINFORCE
# Adapted from https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63

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

# Update Function
def update_policy(policy_network, rewards, log_probs):
    GAMMA = 0.9
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0 
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)
        
    discounted_rewards = torch.tensor(discounted_rewards)
    # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards
    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)
    
    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()


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

    ACTION_DIM = env.action_space.shape[0]
    STATE_DIM = env.observation_space["state"].shape[0]

    # create NN for training policy
    policy_net = PolicyNetwork(STATE_DIM, ACTION_DIM, 128)
    
    if args.vis: env.render(use_imshow=True)
    
    max_episode_num = 5000
    max_steps = 1000
    numsteps = []
    avg_numsteps = []
    all_rewards = []
    PATH="kitchen_model.pt"

    
    for episode in range(max_episode_num):
        env.cam_0.start_recording()


        print("Episode ", episode)
        # reset the state
        state = env.reset()["state"]
        log_probs = []
        rewards = []

        # walk through each step
        for steps in range(max_steps):
            # print("Step ", steps)
            
            # if the mug falls off the table, end the episode
            if env.mug.get_pos()[1] < -0.23:
                break
            
            # get next action, state 
            action, log_prob = policy_net.get_action(state)
            new_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            # if task is successfully completed
            if done:
                # save successful task execution as a video
                # SAVE_FILENAME = 'kitchen_task_recording' + '.mp4'
                # env.cam_0.stop_recording(save_to_filename='kitchen_task_recording.mp4', fps=60)
                env.cam_0.stop_recording(fps=60)
                env.cam_0.start_recording()

                update_policy(policy_net, rewards, log_probs)
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards)) # TODO: the total reward is always 1.0. Why???
                print("ALL REWARDS:", all_rewards)
                if episode % 1 == 0:
                    print("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3),  np.round(np.mean(all_rewards[-10:]), decimals = 3), steps))
                break
        
            
            state = new_state["state"]

        env.cam_0.stop_recording(save_to_filename='scrap.mp4', fps=60)

        # Save the model (https://pytorch.org/tutorials/beginner/saving_loading_models.html)
        torch.save(policy_net, PATH)

    y = env.successes
    x = [sim for sim in range(0, y)]
    plt.plot(x, y, marker='o', linestyle='-')

    # Display the graph
    plt.show()     

    # action structure:
    # action:  {'action': array([ 0.41947876, -1.5054056 ,  1.70436717, -1.3545354 ,  1.62811608,-1.63153025,  0.50537103])}
