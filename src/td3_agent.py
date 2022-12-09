import torch
import random

import numpy as np
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, store_spikes=False, simtime=10):
        self.store_spikes = store_spikes

        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        if self.store_spikes:
            self.state_spikes_memory = np.zeros((self.mem_size, input_shape[0], simtime))
            self.new_state_spikes_memory = np.zeros((self.mem_size, input_shape[0], simtime))

    def store_transition(self, state, action, reward, state_, done, state_spikes=None,
                         state_spikes_=None):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        if self.store_spikes:
            self.state_spikes_memory[index] = state_spikes
            self.new_state_spikes_memory[index] = state_spikes_

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        if self.store_spikes:
            states_spikes = self.state_spikes_memory[batch]
            states_spikes_ = self.new_state_spikes_memory[batch]
            return states, states_spikes, actions, rewards, states_, states_spikes_, dones

        return states, actions, rewards, states_, dones


class Agent():
    def __init__(self, actor, critic_1, critic_2, target_actor, target_critic_1, target_critic_2,
                 input_dims, tau, env, n_timesteps, result_dir, gamma=0.99, update_actor_interval=2,
                 update_target_interval=2, warmup=1000, learning_starts=1000, n_actions=2,
                 buffer_size=1000000, batch_size=100, noise=0.1, seed=0, pop_coding=False,
                 mutually_exclusive=False, pop_size=2, obs_range=[(-1,1)], spiking=False,
                 two_neuron=False, normalize=False, spiking_critic=False, simtime=20,
                 encoding='current', store_spikes=False):
        """
        :param alpha: actor network learning rate
        :param beta: critic network learning rate
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.input_dims = input_dims
        self.tau = tau
        self.env = env
        self.n_timesteps = n_timesteps
        self.result_dir = result_dir
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.max_obs = env.observation_space.high

        for i in range(len(self.max_obs)):
            if self.max_obs[i] == np.inf:
                self.max_obs[i] = 1
        self.gamma = gamma
        self.store_spikes = store_spikes
        self.memory = ReplayBuffer(buffer_size, input_dims, n_actions, store_spikes=store_spikes,
                                   simtime=simtime)
        self.batch_size = batch_size
        self.episode_counter = 0
        self.learn_step_counter = 0
        self.policy_learn_step_counter = 0
        self.time_step = 0
        self.warmup = warmup
        self.learning_starts = learning_starts
        self.n_actions = n_actions
        self.update_actor_interval = update_actor_interval
        self.update_target_interval = update_target_interval
        self.spiking = spiking
        self.spiking_critic = spiking_critic
        self.two_neuron = two_neuron
        self.normalize = normalize
        self.simtime = simtime
        self.encoding = encoding

        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.target_actor = target_actor
        self.target_critic_1 = target_critic_1
        self.target_critic_2 = target_critic_2

        self.chosen_actions = []

        self.noise = noise
        self.update_network_parameters(tau=1)

        self.actor_output = []
        self.critic_1_output = []
        self.critic_2_output = []

        self.pop_coding = pop_coding
        self.mutually_exclusive = mutually_exclusive
        self.pop_size = pop_size
        self.obs_range = obs_range

        if self.pop_coding:
            self.pop_disp = [(i[1] - i[0])/(pop_size + 1) for i in obs_range]
            self.pop_means = []
            for i in range(int(input_dims[0]/pop_size)):
                self.pop_means.append([])
                start = obs_range[i][0]
                for j in range(pop_size):
                    self.pop_means[-1].append(start + self.pop_disp[i])
                    start += self.pop_disp[i]

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = torch.tensor(np.random.normal(scale=self.noise, size=self.n_actions),
                              device=device)
        else:
            if self.normalize:
                observation = self.normalize_state(observation)
                state = observation.clone().to(device)
            else:
                state = torch.tensor(observation, dtype=torch.float).clone().to(device)
            if self.spiking:
                if self.encoding == 'poisson':
                    state = self.generate_poisson_input(state.to('cpu'))
                state = state.unsqueeze(0).to(device)
                mu = self.actor.forward(state)[0].squeeze(0).to(device)
            else:
                mu = self.actor.forward(state).to(device)

        mu_prime = mu + torch.tensor(np.random.normal(scale=self.noise), dtype=torch.float,
                                     device=device).to(device)

        mu_prime = torch.clamp(mu_prime*self.max_action[0], self.min_action[0], self.max_action[0])
        self.time_step += 1

        return mu_prime.cpu().detach().numpy()

    def learn(self):
        if self.memory.mem_counter < self.batch_size or self.time_step < self.learning_starts:
            return
        #if self.encoding == 'poisson':
        #    state, state_spikes, action, reward, state_, state_spikes_, done =\
        #        self.memory.sample_buffer(self.batch_size)
            #state_spikes = torch.tensor(state_spikes, dtype=torch.float).to(device)
            #state_spikes_ = torch.tensor(state_spikes_, dtype=torch.float).to(device)
        #else:
        state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)

        state = torch.tensor(state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.float).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        state_ = torch.tensor(state_, dtype=torch.float).to(device)
        done = torch.tensor(done).to(device)

        if self.normalize:
            state = self.normalize_state(state.to('cpu')).float().to(device)
            state_ = self.normalize_state(state_.to('cpu')).float().to(device)

        if self.spiking:
            if self.encoding == 'poisson':
                state_spikes_ = self.generate_poisson_input(state_.to('cpu')).to(device)
                target_actions = self.target_actor.forward(state_spikes_)[0].squeeze(0).to(device)
            else:
                target_actions = self.target_actor.forward(state_)[0].squeeze(0).to(device)
        else:
            target_actions = self.target_actor.forward(state_)

        if self.spiking_critic:
            q1 = self.critic_1.forward(state, action)[0]
            q2 = self.critic_2.forward(state, action)[0]
        else:
            q1 = self.critic_1.forward(state, action)
            q2 = self.critic_2.forward(state, action)
        target_actions = target_actions + torch.clamp(torch.tensor(np.random.normal(scale=0.2)),
                                                      -0.5, 0.5)
        target_actions = torch.clamp(target_actions, self.min_action[0], self.max_action[0])

        if self.spiking_critic:
            q1_ = self.target_critic_1.forward(state_, target_actions)[0]
            q2_ = self.target_critic_2.forward(state_, target_actions)[0]
        else:
            q1_ = self.target_critic_1.forward(state_, target_actions)
            q2_ = self.target_critic_2.forward(state_, target_actions)

        q1_[done] = 0.0
        q2_[done] = 0.0

        critic_value_ = torch.min(q1_.view(-1), q2_.view(-1)).detach()
        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward(retain_graph=True)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_counter += 1

        if self.learn_step_counter % self.update_actor_interval == 0:
            self.policy_learn_step_counter += 1
            self.actor.optimizer.zero_grad()
            if self.spiking_critic:
                actor_q1_loss = \
                    self.critic_1.forward(state, self.actor.forward(state)[0].squeeze(0))[0]
            elif self.spiking:
                if self.encoding == 'poisson':
                    state_spikes = self.generate_poisson_input(state.to('cpu')).to(device)
                    actor_q1_loss = \
                        self.critic_1.forward(state, self.actor.forward(state_spikes)[0].squeeze(0))
                else:
                    actor_q1_loss = \
                        self.critic_1.forward(state, self.actor.forward(state)[0].squeeze(0))
            else:
                actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
            actor_loss = -torch.mean(actor_q1_loss)
            actor_loss.backward(retain_graph=True)
            self.actor.optimizer.step()

            if self.learn_step_counter % self.update_target_interval == 0:
                self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # update actor params
        if self.spiking:
            actor_state_dict = [d.clone() for d in self.actor.state_dict()[0]]
            target_actor_state_dict = [d.clone() for d in self.target_actor.state_dict()[0]]
            new_state_dict = [[tau*a + (1 - tau)*ta for a, ta in zip(actor_state_dict,
                                                                     target_actor_state_dict)]]
            self.target_actor.load_state_dict(new_state_dict)
        else:
            actor_params = self.actor.named_parameters()
            target_actor_params = self.target_actor.named_parameters()
            actor = dict(actor_params)
            target_actor = dict(target_actor_params)

            for name in actor:
                actor[name] = tau*actor[name].clone() + (1 - tau)*target_actor[name].clone()

            self.target_actor.load_state_dict(actor)

        # update critic params
        if self.spiking_critic:
            critic_1_state_dict = [d.clone() for d in self.critic_1.state_dict()[0]]
            target_critic_1_state_dict = [d.clone() for d in self.target_critic_1.state_dict()[0]]
            new_state_dict = [[tau * a + (1 - tau) * ta for a, ta in
                               zip(critic_1_state_dict, target_critic_1_state_dict)]]
            self.target_critic_1.load_state_dict(new_state_dict)

            critic_2_state_dict = [d.clone() for d in self.critic_2.state_dict()[0]]
            target_critic_2_state_dict = [d.clone() for d in self.target_critic_2.state_dict()[0]]
            new_state_dict = [[tau * a + (1 - tau) * ta for a, ta in
                               zip(critic_2_state_dict, target_critic_2_state_dict)]]
            self.target_critic_2.load_state_dict(new_state_dict)
        else:
            critic_1_params = self.critic_1.named_parameters()
            critic_2_params = self.critic_2.named_parameters()
            target_critic_1_params = self.target_critic_1.named_parameters()
            target_critic_2_params = self.target_critic_2.named_parameters()
            critic_1 = dict(critic_1_params)
            critic_2 = dict(critic_2_params)
            target_critic_1 = dict(target_critic_1_params)
            target_critic_2 = dict(target_critic_2_params)

            for name in critic_1:
                critic_1[name] = tau * critic_1[name].clone() + (1 - tau) * target_critic_1[
                    name].clone()

            for name in critic_2:
                critic_2[name] = tau * critic_2[name].clone() + (1 - tau) * target_critic_2[
                    name].clone()

            self.target_critic_1.load_state_dict(critic_1)
            self.target_critic_2.load_state_dict(critic_2)

    def get_active_neurons(self, state):
        intervals = []
        neuron_idxs = np.zeros_like(state)
        for i in self.obs_range:
            intervals.append((i[1] - i[0]) / self.pop_size)

        for i in range(len(state)):
            threshold = self.obs_range[i][0]
            for k in range(self.pop_size):
                neuron_idx = k
                neuron_idxs[i] = int(neuron_idx)
                if state[i] < threshold + intervals[i]:
                    break
                else:
                    threshold += intervals[i]
        neuron_idxs = neuron_idxs.reshape(state.shape)
        return neuron_idxs

    def get_mutually_exclusive_pop_input(self, state):
        neuron_idxs = self.get_active_neurons(state)
        pop_observation = np.zeros((state.shape[0] * self.pop_size))
        idx = 0
        for i in range(len(state)):
            pop_idx = int(idx + neuron_idxs[i])
            pop_observation[pop_idx] = abs(state[i])
            idx += self.pop_size
        return pop_observation

    def get_population_input(self, state):
        pop_input = []
        for i in range(len(state)):
            for j in range(self.pop_size):
                a_es = np.exp(-0.5*((state[i] - self.pop_means[i][j])/self.pop_disp[i])**2)
                pop_input.append(a_es)

        return pop_input

    def generate_poisson_input(self, state):
        return torch.bernoulli(
            torch.tile(state[..., None], [1] * len(state.shape) + [self.simtime])
        )

    def normalize_state(self, state):
        if self.two_neuron:
            two_neuron_max_obs = np.array([val for val in self.max_obs for _ in (0, 1)])
            return torch.tensor(state/two_neuron_max_obs, dtype=torch.float)

        return torch.tensor(state/self.max_obs, dtype=torch.float)

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

    def update_max_obs(self, state):
        self.max_obs = np.maximum(self.max_obs, np.abs(state))

    def train_agent(self):
        best_average = -np.inf
        best_average_after = np.inf
        reward_history = []
        smoothed_scores = []

        while self.learn_step_counter < self.n_timesteps + 1:
            self.episode_counter += 1
            observation = self.env.reset()
            self.update_max_obs(observation)
            if self.two_neuron:
                observation = self.transform_state(observation)
            done = False
            score = 0
            while not done:
                action = self.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                self.update_max_obs(observation_)
                if self.two_neuron:
                    observation_ = self.transform_state(observation_)
                if self.spiking:
                    observation_ = observation_.reshape(self.input_dims)
                self.memory.store_transition(observation, action, reward, observation_, done)
                score += reward
                observation = observation_
                self.learn()

            reward_history.append(score)
            avg_score = np.mean(reward_history[-100:])
            smoothed_scores.append(avg_score)

            if avg_score > best_average:
                best_average = avg_score
                best_average_after = self.episode_counter
                #self.save_models(self.result_dir)

            print('Episode: ', self.episode_counter, 'training steps: ',
                  self.learn_step_counter, 'score: %.1f' % score,
                  'Average Score: %.1f' % avg_score, end='\r')

            if self.episode_counter % 100 == 0:
                print("\rEpisode: ", self.episode_counter, 'training steps: ', self.learn_step_counter,
                      "Average Score: %.2f" % avg_score)
                self.save_models(self.result_dir, self.episode_counter)

        print('Best 100 episode average: ', best_average, ' reached at episode ',
              best_average_after, '. Model saved in folder best.')
        return smoothed_scores, reward_history, best_average, best_average_after

    def save_models(self, result_dir, episode_num):
        self.actor.save_checkpoint(result_dir, episode_num)
        self.target_actor.save_checkpoint(result_dir, episode_num)
        self.critic_1.save_checkpoint(result_dir)
        self.critic_2.save_checkpoint(result_dir)
        self.target_critic_1.save_checkpoint(result_dir)
        self.target_critic_2.save_checkpoint(result_dir)

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
