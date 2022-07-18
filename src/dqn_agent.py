import sys
import gym
import torch
import random

import numpy as np
import torch.nn.functional as F

from model import DSNN
from collections import namedtuple, deque


sys.path.append('../')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward",
                                                                "next_state", "done"])
        random.seed(seed)
        np.random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).\
            float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).\
            long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).\
            float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).
                                 astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, env, policy_net, target_net, architecture, batch_size, memory_size, gamma,
                 eps_start, eps_end, eps_decay, update_every, target_update_frequency, optimizer,
                 learning_rate, num_episodes, max_steps, i_run, result_dir, seed, tau,
                 spiking=False, two_neuron=False):

        self.env = gym.make(env)
        self.env.seed(seed)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.policy_net = policy_net
        self.target_net = target_net

        self.architecture = architecture
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.update_every = update_every
        self.target_update_frequency = target_update_frequency
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.i_run = i_run
        self.result_dir = result_dir
        self.tau = tau
        self.spiking = spiking
        self.random = random
        self.two_neuron = two_neuron

        # Initialize Replay Memory
        self.memory = ReplayBuffer(self.memory_size, self.batch_size, seed)

        # Initialize time step
        self.t_step = 0
        self.t_step_total = 0

    def select_action(self, state, eps=0.):
        state = torch.from_numpy(state)
        state = state.unsqueeze(0).to(device)
        if random.random() > eps:
            with torch.no_grad():
                if self.spiking:
                    final_layer_values = self.policy_net.forward(state.float())[0].\
                        cpu().data.numpy()
                    return np.argmax(final_layer_values)
                else:
                    return np.argmax(self.policy_net.forward(state.float())[0].cpu().data.numpy())
        else:
            return random.choice(np.arange(self.architecture[-1]))

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.optimize_model(experiences)

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        if self.spiking:
            Q_targets_next = self.target_net.forward(next_states)[0].detach().max(1)[0].unsqueeze(1)
        else:
            Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next*(1 - dones))

        # Get expected Q values from local model
        if self.spiking:
            Q_expected = self.policy_net.forward(states)[0].gather(1, actions)
        else:
            Q_expected = self.policy_net.forward(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)

        self.optimizer.step()
        if self.t_step_total % self.target_update_frequency == 0:
            self.soft_update()

    def soft_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def transform_state(self, state):
        state_ = []
        for i in state:
            if i > 0:
                state_.append(i)
                state_.append(0)
            else:
                state_.append(0)
                state_.append(abs(i))
        return np.array(state_)

    def train_agent(self):
        best_average = -np.inf
        best_average_after = np.inf
        scores = []
        smoothed_scores = []
        scores_window = deque(maxlen=100)
        eps = self.eps_start

        for i_episode in range(1, self.num_episodes + 1):
            state = self.env.reset()
            if self.two_neuron:
                state = self.transform_state(state)
            score = 0
            done = False
            while not done:
                self.t_step_total += 1
                action = self.select_action(state, eps)
                next_state, reward, done, _ = self.env.step(action)
                if self.two_neuron:
                    next_state = self.transform_state(next_state)
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                eps = max(self.eps_end, self.eps_decay * eps)
                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            smoothed_scores.append(np.mean(scores_window))

            if smoothed_scores[-1] > best_average:
                best_average = smoothed_scores[-1]
                best_average_after = i_episode
                if self.spiking:
                    torch.save(self.policy_net.state_dict(),
                               self.result_dir + '/checkpoint_DSQN_{}.pt'.format(self.i_run))
                else:
                    torch.save(self.policy_net.state_dict(),
                               self.result_dir + '/checkpoint_DQN_{}.pt'.format(self.i_run))

            print("Episode {}\tAverage Score: {:.2f}\t Epsilon: {:.2f}".
                  format(i_episode, np.mean(scores_window), eps), end='\r')

            if i_episode % 100 == 0:
                print("\rEpisode {}\tAverage Score: {:.2f}".
                      format(i_episode, np.mean(scores_window)))

        print('Best 100 episode average: ', best_average, ' reached at episode ',
              best_average_after, '. Model saved in folder best.')
        return smoothed_scores, scores, best_average, best_average_after


def evaluate_agent(policy_net, env, num_episodes, max_steps, gym_seeds, epsilon=0):
    """

    """
    rewards = []

    for i_episode in range(num_episodes):
        env.seed(int(gym_seeds[i_episode]))
        env._max_episode_steps = max_steps
        state = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        total_reward = 0
        for t in range(max_steps):
            if random.random() >= epsilon:
                final_layer_values = policy_net.forward(state.float())[0].cpu().data.numpy()
                action = np.argmax(final_layer_values)
            else:
                action = random.randint(0, env.action_space.n - 1)

            observation, reward, done, _ = env.step(action)
            state = torch.from_numpy(observation).float().unsqueeze(0).to(device)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        print("Episode: {}".format(i_episode), end='\r')

    return rewards
