import torch
import numpy as np
from copy import deepcopy

import site
site.addsitedir('../src/')

default_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def evaluate_policy(policy_net, env, n_evaluations, seeds):
    """ Evaluate a policy on a cartpole environment.

    Arguments:
        - policy_net: policy to evaluate. Should have four inputs and 2 outputs.
        - env: Environment object to use for evaluation.
        - n_evaluations: Number of evaluation runs.
        - seeds: Seeds for the environment. Should have at least 'n_evaluations' entries.

    Returns:
        List of rewards.
    """
    eval_rewards = []
    for i in range(n_evaluations):
        env.seed(int(seeds[i]))

        state = env.reset()
        reward = 0
        done = False

        while not done:
            inputs = torch.from_numpy(state).float()
            mem_result, _ = policy_net(inputs, rstdp_state=(None, None, None))
            action = torch.argmax(mem_result)
            state, r, done, _ = env.step(action.item())
            reward += r
        eval_rewards.append(reward)

    return eval_rewards


def rstdp_train_cartpole(policy_net, env, max_reward, num_episodes, n_evaluations, max_steps, rstdp_seeds, evaluation_seeds):
    """ Train policy on cartpole environment with RSTDP.

    Arguments:
        - policy_net: Policy to train.
        - env: Environment object to train on.
        - max_reward: Maximum achievable reward in 'env'. Used for weight update calculation.
        - num_episodes: Number of training episodes.
        - n_evaluations: Number of evaluations for stopping/saving criterion.
        - max_steps: Maximum number of steps in 'env'.
        - rstdp_seeds: Environment seeds for rstdp-training.
        - evaluations_seeds: Environment seeds for evaluation.

    Returns:
        - Trained weights.
        - Rewards achieved during training.
    """
    env._max_episode_steps = max_steps
    
    best_reward = -np.inf
    best_episode = -1
    best_weights = None
    
    rewards = []
    
    for i_episode in range(num_episodes):
        env.seed(int(rstdp_seeds[i_episode]))
        
        e_trace = None
        
        state = env.reset()
        total_reward = 0
        for t in range(max_steps):
            inputs = torch.from_numpy(state).float()
            final_layer_values, rstdp_out = policy_net.forward(inputs, (e_trace, None, None))
            final_layer_values = final_layer_values.cpu().data.numpy()
            e_trace = rstdp_out[0][0]

            action = np.argmax(final_layer_values)
            state, reward, done, _ = env.step(action)

            total_reward += reward
            if done:
                break

        rewards.append(total_reward)
                
        # RSTDP update
        delta_rstdp = -e_trace * (1.0 - total_reward / max_reward)
        policy_net.l2.weights += delta_rstdp[0]
        
        eval_rewards = evaluate_policy(policy_net, env, n_evaluations, evaluation_seeds)
        avg_eval_reward = np.mean(eval_rewards)
        
        print("Episode: {:4d} -- Reward: {:7.2f} -- Best reward: {:7.2f} in episode {:4d}"\
                .format(i_episode, avg_eval_reward, best_reward, best_episode), end='\r')
        
        if avg_eval_reward > best_reward:
            best_reward = avg_eval_reward
            best_episode = i_episode
            best_weights = deepcopy(policy_net.state_dict())
            
        if best_reward >= max_reward:
            break
            
    print('\nBest individual stored after episode {:d} with reward {:6.2f}'.format(best_episode, best_reward))
    print()
    return best_weights, rewards


def rstdp_train_acrobot(policy_net, env, min_reward, num_episodes, n_evaluations, max_steps, rstdp_seeds, evaluation_seeds):
    """ Train policy on acrobot environment with RSTDP.

    Arguments:
        - policy_net: Policy to train.
        - env: Environment object to train on.
        - max_reward: Maximum achievable reward in 'env'. Used for weight update calculation.
        - num_episodes: Number of training episodes.
        - n_evaluations: Number of evaluations for stopping/saving criterion.
        - max_steps: Maximum number of steps in 'env'.
        - rstdp_seeds: Environment seeds for rstdp-training.
        - evaluations_seeds: Environment seeds for evaluation.

    Returns:
        - Trained weights.
        - Rewards achieved during training.
    """
    env._max_episode_steps = max_steps
    
    best_reward = -np.inf
    best_episode = -1
    best_weights = None
    
    rewards = []
    
    for i_episode in range(num_episodes):
        env.seed(int(rstdp_seeds[i_episode]))
        
        e_trace = None
        
        state = env.reset()
        total_reward = 0
        for t in range(max_steps):
            inputs = torch.from_numpy(state).float()
            final_layer_values, rstdp_out = policy_net.forward(inputs, (e_trace, None, None))
            final_layer_values = final_layer_values.cpu().data.numpy()
            e_trace = rstdp_out[0][0]

            action = np.argmax(final_layer_values)
            state, reward, done, _ = env.step(action)

            total_reward += reward
            if done:
                break

        rewards.append(total_reward)
                
        # RSTDP update
        delta_rstdp = e_trace * (total_reward/min_reward)
        policy_net.l2.weights += delta_rstdp[0]
        
        eval_rewards = evaluate_policy(policy_net, env, n_evaluations, evaluation_seeds)
        avg_eval_reward = np.mean(eval_rewards)
        
        print("Episode: {:4d} -- Reward: {:7.2f} -- Best reward: {:7.2f} in episode {:4d}"\
                .format(i_episode, avg_eval_reward, best_reward, best_episode), end='\r')
        
        if avg_eval_reward > best_reward:
            best_reward = avg_eval_reward
            best_episode = i_episode
            best_weights = deepcopy(policy_net.state_dict())
            
        if best_reward >= -100:
            break
            
    print('\nBest individual stored after episode {:d} with reward {:6.2f}'.format(best_episode, best_reward))
    print()
    return best_weights, rewards