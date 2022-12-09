import os
import torch
import random

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        # Only for inhibitory spikes
        #out[input < 0] = -1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


class DSNN(nn.Module):
    def __init__(self, architecture, seed, alpha, beta, weight_scale, batch_size, threshold,
                 simulation_time, learning_rate, reset_potential=0):
        """

        """
        self.architecture = architecture

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.alpha = alpha
        self.beta = beta
        self.weight_scale = weight_scale
        self.batch_size = batch_size
        self.threshold = threshold
        self.simulation_time = simulation_time
        self.reset_potential = reset_potential

        self.spike_fn = SurrGradSpike.apply

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Initialize the network weights
        self.weights = []
        for i in range(len(architecture) - 1):
            self.weights.append(torch.empty((self.architecture[i], self.architecture[i + 1]),
                                            device=device, dtype=torch.float, requires_grad=True))
            torch.nn.init.normal_(self.weights[i], mean=0.0,
                                  std=self.weight_scale/np.sqrt(self.architecture[i]))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, inputs):
        syn = []
        mem = []
        spk_count = []

        for l in range(0, len(self.weights)):
            syn.append(torch.zeros((self.batch_size, self.weights[l].shape[1]), device=device,
                                   dtype=torch.float))
            mem.append(torch.zeros((self.batch_size, self.weights[l].shape[1]), device=device,
                                   dtype=torch.float))

        # Here we define two lists which we use to record the membrane potentials and output spikes
        mem_rec = []
        spk_rec = []

        # Here we loop over time
        for t in range(self.simulation_time):
            # append the new timestep to mem_rec and spk_rec
            mem_rec.append([])
            spk_rec.append([])

            if t == 0:
                for l in range(len(self.weights)):
                    mem_rec[-1].append(mem[l])
                    spk_rec[-1].append(mem[l])
                continue

            # We take the input as it is, multiply is by the weights, and we inject the outcome
            # as current in the neurons of the first hidden layer
            input = inputs.detach().clone()

            # loop over layers
            for l in range(len(self.weights)):
                if l == 0:
                    h = torch.einsum("ab,bc->ac", [input, self.weights[0]])
                    new_syn = 0 * syn[l] + h
                else:
                    h = torch.einsum("ab,bc->ac", [spk_rec[-1][l - 1], self.weights[l]])
                    new_syn = self.alpha * syn[l] + h

                new_mem = self.beta*mem[l] + new_syn

                # calculate the spikes for all layers but the last layer
                if l < (len(self.weights) - 1):
                    mthr = new_mem
                    mthr = mthr - self.threshold
                    out = self.spike_fn(mthr)
                    c = (mthr > 0)
                    new_mem[c] = self.reset_potential
                    spk_rec[-1].append(out)

                mem[l] = new_mem
                syn[l] = new_syn

                mem_rec[-1].append(mem[l])

        # return the final recorded membrane potential in the output layer, all membrane potentials,
        # and spikes
        return mem_rec[-1][-1], mem_rec, spk_rec

    def load_state_dict(self, layers):
        """Method to load weights and biases into the network"""
        weights = layers[0]
        for l in range(0,len(weights)):
            self.weights[l] = weights[l].detach().clone().requires_grad_(True)

    def state_dict(self):
        """Method to copy the layers of the SQN. Makes explicit copies, no references."""
        weights_copy = []
        bias_copy = []
        for l in range(0, len(self.weights)):
            weights_copy.append(self.weights[l].detach().clone())
        return weights_copy, bias_copy

    def parameters(self):
        parameters = []
        for l in range(0, len(self.weights)):
            parameters.append(self.weights[l])

        return parameters


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, architecture, seed):
        """Initialize parameters and build model.
        Params
        ======
            architecture:
        """
        super(QNetwork, self).__init__()
        self.layers = nn.ModuleList()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        for i in range(len(architecture) - 1):
            self.layers.append(nn.Linear(architecture[i], architecture[i + 1]))

    def forward(self, x):
        """Build a network that maps state -> action values."""
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        # no ReLu activation in the output layer
        return self.layers[-1](x)


class TD3ActorDSNN(DSNN):
    def __init__(self, architecture, seed, alpha, beta, weight_scale, batch_size, threshold,
                 simulation_time, learning_rate, name, device, reset_potential=0,
                 random_params=False, std=0.1):
        self.architecture = architecture

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.alpha = alpha
        self.beta = beta
        self.weight_scale = weight_scale
        self.batch_size = batch_size
        self.threshold = threshold
        self.simulation_time = simulation_time
        self.reset_potential = reset_potential
        self.random_params = random_params
        self.std = std

        self.name = name

        self.spike_fn = SurrGradSpike.apply
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Initialize the network weights
        self.weights = []
        for i in range(len(architecture) - 1):
            self.weights.append(torch.empty((self.architecture[i], self.architecture[i + 1]),
                                            device=device, dtype=torch.float, requires_grad=True))
            torch.nn.init.normal_(self.weights[i], mean=0.0,
                                  std=self.weight_scale / np.sqrt(self.architecture[i]))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        if random_params:
            self.randomize_parameters()

    def randomize_parameters(self):
        # Define new random values for alpha, beta, and threhsold
        self.alphas = []
        self.betas = []
        self.thresholds = []
        self.reset_potentials = []

        for i in range(len(self.architecture) - 1):
            self.alphas.append(torch.normal(torch.ones(self.architecture[i + 1])*self.alpha,
                                            self.alpha*self.std).
                               unsqueeze(0).to(self.device))
            self.alphas[i][self.alphas[i] > 1] = 1
            self.alphas[i][self.alphas[i] < 0] = 0
            self.betas.append(torch.normal(torch.ones(self.architecture[i + 1])*self.beta,
                                           self.beta*self.std).
                              unsqueeze(0).to(self.device))
            self.betas[i][self.betas[i] > 1] = 1
            self.betas[i][self.betas[i] < 0] = 0

            self.thresholds.append(
                torch.normal(torch.ones(self.architecture[i + 1]) * self.threshold,
                             self.threshold*self.std).to(self.device))
            self.thresholds[i][self.thresholds[i] < 0] = 0

            self.reset_potentials.append(
                torch.normal(torch.ones(self.architecture[i + 1])*self.reset_potential,
                             self.std).to(self.device))

    def forward(self, inputs):
        syn = []
        mem = []

        for l in range(0, len(self.weights)):
            syn.append(torch.zeros((self.batch_size, self.weights[l].shape[1]), device=device,
                                   dtype=torch.float))
            mem.append(torch.zeros((self.batch_size, self.weights[l].shape[1]), device=device,
                                   dtype=torch.float))

        # Here we define two lists which we use to record the membrane potentials and output spikes
        mem_rec = []
        spk_rec = []

        # Here we loop over time
        for t in range(self.simulation_time):
            # append the new timestep to mem_rec and spk_rec
            mem_rec.append([])
            spk_rec.append([])

            if t == 0:
                for l in range(len(self.weights)):
                    mem_rec[-1].append(mem[l])
                    spk_rec[-1].append(mem[l])
                continue

            # We take the input as it is, multiply is by the weights, and we inject the outcome
            # as current in the neurons of the first hidden layer
            input = inputs.detach().clone()

            # loop over layers
            for l in range(len(self.weights)):
                if l == 0:
                    h = torch.matmul(input, self.weights[0])
                    new_syn = 0 * syn[l] + h
                elif l == len(self.weights) - 1:
                    h = torch.matmul(spk_rec[-1][l - 1], self.weights[l])
                    new_syn = 0*syn[l] + h
                else:
                    h = torch.matmul(spk_rec[-1][l - 1], self.weights[l])
                    if self.random_params:
                        new_syn = torch.add(torch.mul(self.alphas[l], syn[l]), h)
                    else:
                        new_syn = self.alpha * syn[l] + h

                if l == len(self.weights) - 1:
                    new_mem = 1*mem[l] + new_syn
                else:
                    if self.random_params:
                        new_mem = torch.add(torch.mul(self.betas[l], mem[l]), new_syn)
                    else:
                        new_mem = self.beta*mem[l] + new_syn

                # calculate the spikes for all layers but the last layer (decoding='potential')
                if l < (len(self.weights) - 1):
                    mthr = new_mem
                    if self.random_params:
                        mthr = torch.sub(mthr, self.thresholds[l])
                        reset = self.reset_potential
                    else:
                        mthr = mthr - self.threshold
                        reset = self.reset_potential
                    out = self.spike_fn(mthr)
                    c = (mthr > 0)
                    new_mem[c] = self.reset_potential

                    spk_rec[-1].append(out)

                mem[l] = new_mem
                syn[l] = new_syn

                mem_rec[-1].append(mem[l])

        # return the final recorded membrane potential (len(mem_rec)-1) in the output layer (-1)
        return torch.tanh(mem_rec[-1][-1]), mem_rec, spk_rec

    def save_checkpoint(self, result_dir, episode_num):
        #print('... saving checkpoint ...')
        torch.save(self.state_dict(), result_dir + '/checkpoint_TD3_{}_{}.pt'.format(self.name,
                                                                                     episode_num))


class TD3CriticNetwork(nn.Module):
    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 checkpoint_dir='tmp/td3'):
        super(TD3CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = self.fc1(torch.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoint(self, result_dir):
        torch.save(self.state_dict(), result_dir + '/checkpoint_TD3_{}.pt'.format(self.name))

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))
