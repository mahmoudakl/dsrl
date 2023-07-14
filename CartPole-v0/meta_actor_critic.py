import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--wandb', type=bool, default=True)
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'hidden'])
torch.autograd.set_detect_anomaly(True)

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 2)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []


        action_alphas = torch.zeros(128,2)
        # action_alphas = torch.nn.init.normal_(action_alphas, mean=0.0, std=1.0)
        self.action_alphas = nn.Parameter(action_alphas)
        torch.nn.init.normal_(self.action_alphas)
        # Get random tensor alpha
        # Assign to parameter for self
        value_alphas = torch.zeros(128,1)
        self.value_alphas = nn.Parameter(value_alphas)
        torch.nn.init.normal_(self.value_alphas)

        self.delta_action_weights = torch.zeros(128,2, requires_grad=False, device=device)
        self.delta_value_weights = torch.zeros(128,1, requires_grad=False, device=device)

        self.eff_action_weights = torch.zeros(128,2, requires_grad=False, device=device)
        self.eff_value_weights = torch.zeros(128,1, requires_grad=False, device=device)

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        self.eff_action_weights = self.action_head.weight.t() + torch.mul(self.action_alphas,self.delta_action_weights)
        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(torch.matmul(self.eff_action_weights.t(), x) + self.action_head.bias, dim=-1)


        # critic: evaluates being in the state s_t
        self.eff_value_weights = self.value_head.weight.t() + self.value_alphas * self.delta_value_weights
        state_values = torch.matmul(self.eff_value_weights.t(), x) + self.value_head.bias

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t

        self.delta_value_weights.detach_()
        self.delta_action_weights.detach_()

        return action_prob, state_values, x

    def restart(self):

        self.delta_action_weights = torch.zeros(128,2, requires_grad=False, device=device)
        self.delta_value_weights = torch.zeros(128,1, requires_grad=False, device=device)

        self.eff_action_weights = torch.zeros(128,2, requires_grad=False, device=device)
        self.eff_value_weights = torch.zeros(128,1, requires_grad=False, device=device)  




eps = np.finfo(np.float32).eps.item()


def select_action(state, model):
    state = torch.from_numpy(state).float().to(device=device)
    probs, state_value, z = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value, z))

    # the action to take (left or right)
    return action.item(), probs


def finish_episode(model, optimizer):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value, z), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(device=device)))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]
    model.delta_action_weights.detach_()
    model.delta_value_weights.detach_()
    model.eff_action_weights.detach_()
    model.eff_value_weights.detach_()

meta_td = True
# value_upd = True
# td_learning_rate = 0.001
# trace_decay = 0.7













def online_ep(env, config, model):

    td_learning_rate = config.td_lr
    trace_decay = config.decay
    value_upd = config.value_upd
    time_R = config.time_R
    proba = config.proba
    mov_av = config.mov_av

    model.restart()
    # reset environment and episode reward
    state = env.reset()
    ep_reward = 0

    # for each episode, only run 9999 steps so that we don't
    # infinite loop while learning
    td_error = torch.zeros(1,requires_grad=False)
    td_errors = []
    hidden_trace = torch.zeros((128,), device=device)
    action_trace = torch.zeros((2,), device=device)
    value = torch.full((1,),-abs(state[2]/0.2), device=device)
    td_error_MA = 0        
    for t in range(1, 10000):

        
        # select action from policy
        action, probs = select_action(state, model)

        # the action to take (left or right)
        

        # DELTA UPDATE TO EFFECTIVE WEIGHTS
        if meta_td:
            with torch.no_grad():

                action_prob, new_value, z = model.saved_actions[-1]
                # new_value = state_value.detach_()
                if time_R:
                    reward = 1
                else:
                    reward = -abs(state[2]/0.2)
                
                ### GET Weight UPDATE ###

                hidden_trace = trace_decay*hidden_trace + z.detach()

                hidden_trace.detach_()
                action_oh = F.one_hot(torch.full((1,),action), num_classes=2).to(device=device)
                if proba:
                    action_trace = trace_decay*action_trace + (action_oh - probs.detach())
                else:
                    action_trace = trace_decay*action_trace + action_oh

                action_trace.detach_()
                td_error = reward + args.gamma*new_value.detach() - value.detach()

                if mov_av:
                    td_errors.append(td_error.cpu().numpy())
                    td_error_MA = np.mean(td_errors[-11:])

                    td_error = td_error - td_error_MA

                td_error.detach_()


                
                if value_upd:
                    value_learning_value = torch.einsum("z,v->zv", [hidden_trace, td_error])
                    value_learning_value.detach_()
                    val_weights = model.delta_value_weights.detach()
                    model.delta_value_weights = val_weights + (td_learning_rate*value_learning_value)

                action_learning_value = torch.einsum("z,ba->za", [hidden_trace, action_trace])
                action_learning_value = torch.einsum("za,v->za", [action_learning_value, td_error]).detach()
                action_learning_value.detach_()
                act_weights = model.delta_action_weights.detach()
                model.delta_action_weights = act_weights + (td_learning_rate*action_learning_value)
                
                model.delta_value_weights.detach_()
                model.delta_action_weights.detach_()

                value = new_value.detach()

        # take the action
        state, reward, done, _ = env.step(action)

        if args.render:
            env.render()

        model.rewards.append(reward)
        ep_reward += reward
    

        if done:
            break

    return ep_reward





def main():

    model = Policy().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=3e-2)

    env = gym.make('CartPole-v0')
    env.seed(args.seed)
    env.reset()
    torch.manual_seed(args.seed)
    env._max_episode_steps = 500

    if args.wandb:
        wandb.init() #, config=vars(args))
        # wandb.gym.monitor()


    config = wandb.config

    # td_learning_rate = config.td_lr
    # trace_decay = config.decay
    # value_upd = config.value_upd
    # time_R = config.time_R
    # proba = config.proba
    # mov_av = config.mov_av


    all_rewards = []
    running_reward = 10

    # run infinitely many episodes
    for i_episode in count(1):

        ep_reward = online_ep(env, config, model)
        
        # model.restart()
        # # reset environment and episode reward
        # state = env.reset()
        # ep_reward = 0

        # # for each episode, only run 9999 steps so that we don't
        # # infinite loop while learning
        # td_error = torch.zeros(1,requires_grad=False)
        # td_errors = []
        # hidden_trace = torch.zeros((128,), device=device)
        # action_trace = torch.zeros((2,), device=device)
        # value = torch.full((1,),-abs(state[2]/0.2), device=device)
        # td_error_MA = 0        
        # for t in range(1, 10000):

            
        #     # select action from policy
        #     action, probs = select_action(state)

        #     # the action to take (left or right)
            

        #     # DELTA UPDATE TO EFFECTIVE WEIGHTS
        #     if meta_td:
        #         with torch.no_grad():

        #             action_prob, new_value, z = model.saved_actions[-1]
        #             # new_value = state_value.detach_()
        #             if time_R:
        #                 reward = 1
        #             else:
        #                 reward = -abs(state[2]/0.2)
                    
        #             ### GET Weight UPDATE ###

        #             hidden_trace = trace_decay*hidden_trace + z.detach()

        #             hidden_trace.detach_()
        #             action_oh = F.one_hot(torch.full((1,),action), num_classes=2).to(device=device)
        #             if proba:
        #                 action_trace = trace_decay*action_trace + (action_oh - probs.detach())
        #             else:
        #                 action_trace = trace_decay*action_trace + action_oh

        #             action_trace.detach_()
        #             td_error = reward + args.gamma*new_value.detach() - value.detach()

        #             if mov_av:
        #                 td_errors.append(td_error.cpu().numpy())
        #                 td_error_MA = np.mean(td_errors[-11:])

        #                 td_error = td_error - td_error_MA

        #             td_error.detach_()


                    
        #             if value_upd:
        #                 value_learning_value = torch.einsum("z,v->zv", [hidden_trace, td_error])
        #                 value_learning_value.detach_()
        #                 val_weights = model.delta_value_weights.detach()
        #                 model.delta_value_weights = val_weights + (td_learning_rate*value_learning_value)

        #             action_learning_value = torch.einsum("z,ba->za", [hidden_trace, action_trace])
        #             action_learning_value = torch.einsum("za,v->za", [action_learning_value, td_error]).detach()
        #             action_learning_value.detach_()
        #             act_weights = model.delta_action_weights.detach()
        #             model.delta_action_weights = act_weights + (td_learning_rate*action_learning_value)
                    
        #             model.delta_value_weights.detach_()
        #             model.delta_action_weights.detach_()

        #             value = new_value.detach()

        #     # take the action
        #     state, reward, done, _ = env.step(action)

        #     if args.render:
        #         env.render()

        #     model.rewards.append(reward)
        #     ep_reward += reward
        

        #     if done:
        #         break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        all_rewards.append(running_reward)

        # perform backprop
        finish_episode(model, optimizer)

        if args.wandb:
            wandb.log({'Rewards': running_reward})

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        if running_reward > 500:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to 500 time steps!".format(running_reward))
            break
        elif i_episode>500:
            break 

    env.unwrapped.length *= 3
    env._max_episode_steps = 500
    env.seed(args.seed)

    if args.wandb:
            wandb.log({'Rewards': 0})

    for i in range(10):
        reward = online_ep(env, config, model)
        wandb.log({'Rewards': reward})


        




if args.wandb:
    sweep_config = {
        "method": "bayes",
        "name": "Cartpole_ML_Sweep",
        "metric": {"name": "Rewards", "goal": "maximize"},
        "parameters": {
            "td_lr": {"min": 0.0001, "max": 2.0},
            "decay": {"values": [0.4, 0.5, 0.6, 0.7, 0.8]},
            "value_upd": {"values": [True, False]},
            "time_R": {"values": [True, False]},
            "proba": {"values": [True, False]},
            "mov_av": {"values": [True, False]},

        },
        "early_terminate": {"type": "hyperband", "min_iter": 5},
        "run_cap": 100
    }

    sweep_id = wandb.sweep(sweep_config, project='Cartpole_MT')

    wandb.agent(sweep_id, main)


# if __name__ == '__main__':
#     print('RESET WEIGHTS')
#     main()

 # @TODO: Try: Time_reward, Learning_rate (train?), Action_logits vs (action-logits), Delta only on action_weights, Meta-train trace, No trace for (1-softmax action)