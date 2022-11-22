import gym
import copy
import site
import torch
import random

import numpy as np
import torch.nn as nn

site.addsitedir('..')

from src.dsnn import DSNN
from src.rstdp import RSTDP
from functools import partial
from collections import deque


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
dtype = torch.float

n_evalutaions = 100

evaluation_seeds = np.load('../../seeds/evaluation_seeds.npy')
rstdp_training_seeds = np.load('../../seeds/rstdp_training_seeds.npy')


class Net(nn.Module):
    def __init__(self, alpha, beta, threshold, architecture, weights, tau=16, tau_e=800,
                 A_plus=0.00001, A_minus=0.00001, C=0.01):
        super(Net, self).__init__()

        self.l1 = DSNN(
            state_size=(architecture[1],),
            operation=partial(torch.einsum, 'ab,bc->ac'),
            weights_size=(architecture[0], architecture[1]),
            alpha=alpha,
            beta=beta,
            threshold=threshold
        )
        # Set 'requires_grad' to 'False', since value assignment is not supported otherwise
        self.l1.weights = nn.Parameter(weights[0][0].clone(), requires_grad=False)
        self.l2 = DSNN(
            state_size=(architecture[2],),
            operation=partial(torch.einsum, 'ab,bc->ac'),
            weights_size=(architecture[1], architecture[2]),
            alpha=alpha,
            beta=beta,
            threshold=threshold
        )
        # As above, set 'requires_grad' to 'False', since value assignment is not supported otherwise
        self.l2.weights = nn.Parameter(weights[0][1].clone(), requires_grad=False)
        self.l3 = DSNN(
            state_size=(architecture[3],),
            operation=partial(torch.einsum, 'ab,bc->ac'),
            weights_size=(architecture[2], architecture[3]),
            alpha=alpha,
            beta=beta,
            threshold=threshold,
            is_spiking=False
        )
        self.l3.weights = nn.Parameter(weights[0][2].clone(), requires_grad=False)

        self.rstdp=RSTDP(A_plus=A_plus, A_minus=A_minus, tau_plus=tau, tau_minus=tau, tau_e=tau_e, C=C)

    def forward(self, inputs, rstdp_state=(None, None, None)):

        # capture spikes of layers
        y_l1, _, _ = self.l1(inputs)
        y_l2, _, _ = self.l2(y_l1)
        _, mem_result, _ = self.l3(y_l2)

        # Feed spikes of layers into rstdp function
        rstdp_out = self.rstdp(y_l1, y_l2, rstdp_state)
        return mem_result[0], rstdp_out


def transform_state(state):
    state_ = []
    for i in state:
        if i > 0:
            state_.append(i)
            state_.append(0)
        else:
            state_.append(0)
            state_.append(abs(i))
    return np.array(state_)


def evaluate_cartpole(policy_net, sim_time, masspole=None, force_mag=None, length=None):
    eval_rewards = []
    for i in range(n_evalutaions):
        env = gym.make('CartPole-v0')
        env.seed(int(evaluation_seeds[i]))
        if length is not None:
            env.unwrapped.length = length
        if masspole is not None:
            env.unwrapped.masspole = masspole
        if force_mag is not None:
            env.unwrapped.force_mag = force_mag

        state = env.reset()
        state = transform_state(state)
        reward = 0
        done = False

        while not done:
            inputs = torch.tile(torch.from_numpy(state), (1, sim_time, 1)).float().to(device)
            mem_result, rstdp_out = policy_net(inputs, rstdp_state=(None, None, None))
            action = torch.argmax(mem_result)
            next_state, r, done, _ = env.step(action.item())
            reward += r
            state = transform_state(next_state)
        eval_rewards.append(reward)

    return eval_rewards


def evaluate_acrobot(policy_net, sim_time, mass=None, moi=None, length=None):
    eval_rewards = []
    for i in range(n_evalutaions):
        env = gym.make('Acrobot-v1')
        env.seed(int(evaluation_seeds[i]))
        if length is not None:
            env.unwrapped.LINK_LENGTH_1 = length
            env.unwrapped.LINK_LENGTH_2 = length
        if mass is not None:
            env.unwrapped.LINK_MASS_1 = mass
            env.unwrapped.LINK_MASS_2 = mass
        if moi is not None:
            env.unwrapped.LINK_MOI = moi

        state = env.reset()
        state = transform_state(state)
        reward = 0
        done = False

        while not done:
            inputs = torch.tile(torch.from_numpy(state), (1, sim_time, 1)).float().to(device)
            mem_result, rstdp_out = policy_net(inputs, rstdp_state=(None, None, None))
            action = torch.argmax(mem_result)
            next_state, r, done, _ = env.step(action.item())
            e_trace = rstdp_out[0][0]
            reward += r
            state = transform_state(next_state)
        eval_rewards.append(reward)
        #print('Evaluation --- iteration: {}, reward: {}, reward average: {}'.
        #      format(i, reward, np.mean(eval_rewards)), end='\r')
    return eval_rewards


def run_acrobot_rstdp_experiment_new(weights, alpha, beta, threshold, sim_time, env_name,
                                     A_coeff=1, C=1, tau=16, tau_e=800, length=None, mass=None,
                                     moi=None, n_episodes=20):
    max_reward = -500
    print('r-STDP training ...')

    # train w/ r-STDP
    random_eval_rewards = []
    training_rstdp_rewards = []
    new_weights = []
    best_average_at = []
    avg_weight = []
    min_weight = []
    best_eval = []

    for w in range(len(weights)):
        w_plus = copy.deepcopy(weights[w][0][1])
        w_minus = copy.deepcopy(weights[w][0][1])
        w_plus[w_plus < 0] = 0
        w_minus[w_minus > 0] = 0
        A_plus = torch.mean(w_plus)
        A_minus = torch.abs(torch.mean(w_minus))
        new_weights.append([])
        best_average_at.append(np.inf)
        training_rstdp_rewards.append([])
        scores_window = []
        best_eval.append([])

        best_score = -500
        temp_weights = weights[w][0][1]

        rstdp_policy_net = Net(alpha, beta, threshold, [12, 256, 256, 3], weights[w], tau=tau,
                               tau_e=tau_e, A_plus=A_plus/A_coeff, A_minus=A_minus/A_coeff, C=C)

        for i in range(n_episodes):

            rstdp_policy_net.l2.weights = nn.Parameter(temp_weights.clone(), requires_grad=False)

            eval_rewards = evaluate_acrobot(rstdp_policy_net, sim_time, mass=mass,
                                                length=length, moi=moi)
            if i == 0:
                random_eval_rewards.append(eval_rewards)

            scores_window.append(np.mean(eval_rewards))

            env = gym.make(env_name)
            env.seed(int(rstdp_training_seeds[i]))

            if length is not None:
                env.unwrapped.LINK_LENGTH_1 = length
                env.unwrapped.LINK_LENGTH_2 = length
            if mass is not None:
                env.unwrapped.LINK_MASS_1 = mass
                env.unwrapped.LINK_MASS_2 = mass
            if moi is not None:
                env.unwrapped.LINK_MOI = moi

            state = env.reset()
            state = transform_state(state)
            reward = 0
            e_trace = None
            done = False

            while not done:
                inputs = torch.tile(torch.from_numpy(state), (1, sim_time, 1)).float()
                mem_result, rstdp_out = rstdp_policy_net(inputs, rstdp_state=(e_trace, None, None))
                action = torch.argmax(mem_result)
                next_state, r, done, _ = env.step(action.item())
                e_trace = rstdp_out[0][0]
                reward += r
                state = transform_state(next_state)
            training_rstdp_rewards[w].append(reward)
            print('Training --- iteration: {}, reward: {}, worst_score: {}, best_score: {}'.
                format(i, scores_window[-1], scores_window[0], best_score), end='\r')

            old_weights = rstdp_policy_net.l2.weights
            rstdp_delta = e_trace*((reward/max_reward))#- 0.1)
            rstdp_policy_net.l2.weights += rstdp_delta[0]
            temp_weights = rstdp_policy_net.l2.weights

            new_weights[w].append(copy.deepcopy(old_weights))
            avg_weight.append(torch.mean(torch.abs(old_weights)).item())
            min_weight.append(torch.min(torch.abs(old_weights)).item())

            # save weights if best running average is achieved
            if scores_window[-1] > best_score:
                best_score = scores_window[-1]
                best_eval[w] = eval_rewards
                best_average_at[w] = i

            #if best_score > 1.025*original_scores[w] or scores_window[-1] <= -400:
            if scores_window[-1] <= -400:
                print('')
                break

        print('')

    return random_eval_rewards, best_eval, training_rstdp_rewards, avg_weight,\
           best_average_at


def run_cartpole_rstdp_experiment_new(weights, alpha, beta, threshold, sim_time, env_name,
                                      A_coeff=1, C=1, tau=16, tau_e=800, length=None, masspole=None,
                                      force_mag=None, n_episodes=20):
    max_reward = 200
    print('r-STDP training ...')

    # train w/ r-STDP
    random_eval_rewards = []
    training_rstdp_rewards = []
    new_weights = []
    best_average_at = []
    avg_weight = []
    min_weight = []
    best_eval = []

    for w in range(len(weights)):
        w_plus = copy.deepcopy(weights[w][0][1])
        w_minus = copy.deepcopy(weights[w][0][1])
        w_plus[w_plus < 0] = 0
        w_minus[w_minus > 0] = 0
        A_plus = torch.mean(w_plus)
        A_minus = torch.abs(torch.mean(w_minus))
        new_weights.append([])
        best_average_at.append(np.inf)
        training_rstdp_rewards.append([])
        scores_window = []
        best_eval.append([])

        best_score = 0
        temp_weights = weights[w][0][1]

        rstdp_policy_net = Net(alpha, beta, threshold, [8, 64, 64, 2], weights[w], tau=tau,
                               tau_e=tau_e, A_plus=A_plus/A_coeff, A_minus=A_minus/A_coeff, C=C)

        for i in range(n_episodes):

            rstdp_policy_net.l2.weights = nn.Parameter(temp_weights.clone(), requires_grad=False)

            if i == 0 or training_rstdp_rewards[w][-1] != 200:
                eval_rewards = evaluate_cartpole(rstdp_policy_net, sim_time, masspole=masspole,
                                                length=length, force_mag=force_mag)
            if i == 0:
                random_eval_rewards.append(eval_rewards)

            scores_window.append(np.mean(eval_rewards))

            env = gym.make(env_name)
            env.seed(int(rstdp_training_seeds[i]))

            if length is not None:
                env.unwrapped.length = length
            if masspole is not None:
                env.unwrapped.masspole = masspole
            if force_mag is not None:
                env.unwrapped.force_mag = force_mag

            state = env.reset()
            state = transform_state(state)
            reward = 0
            e_trace = None
            done = False

            while not done:
                inputs = torch.tile(torch.from_numpy(state), (1, sim_time, 1)).float()
                mem_result, rstdp_out = rstdp_policy_net(inputs, rstdp_state=(e_trace, None, None))
                action = torch.argmax(mem_result)
                next_state, r, done, _ = env.step(action.item())
                e_trace = rstdp_out[0][0]
                reward += r
                state = transform_state(next_state)
            training_rstdp_rewards[w].append(reward)
            print('Training --- iteration: {}, reward: {}, best_score: {}'.
                format(i, scores_window[-1], best_score), end='\r')

            old_weights = rstdp_policy_net.l2.weights
            rstdp_delta = -e_trace*(1.0 - reward/max_reward)
            rstdp_policy_net.l2.weights += rstdp_delta[0]
            temp_weights = rstdp_policy_net.l2.weights

            new_weights[w].append(copy.deepcopy(old_weights))
            avg_weight.append(torch.mean(torch.abs(old_weights)).item())
            min_weight.append(torch.min(torch.abs(old_weights)).item())

            # save weights if best average is achieved
            if scores_window[-1] > best_score:
                best_score = scores_window[-1]
                best_eval[w] = eval_rewards
                best_average_at[w] = i

            if best_score == 200:
                print('')
                break

        print('')

    return random_eval_rewards, best_eval, training_rstdp_rewards, avg_weight,\
           best_average_at, new_weights
