import torch
import torch.nn as nn
import random

import numpy as np
from functools import partial

from model import SurrGradSpike
from rstdp import RSTDP

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

class DeltaTDNet(nn.Module):
    def __init__(self, seed, alpha, beta, hidden_weights, actor_weights, value_weights, batch_size, threshold,
                 simulation_time, learning_rate, reset_potential=0):
        """

        """

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if hidden_weights.isinstance(list):
            self.weights = hidden_weights
        else:
            self.weights = [hidden_weights]
        self.actor_weights = actor_weights
        self.value_weights = value_weights

        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size
        self.threshold = threshold
        self.simulation_time = simulation_time
        self.reset_potential = reset_potential

        self.spike_fn = SurrGradSpike.apply

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
                    z = self.spike_fn(mthr)
                    c = (mthr > 0)
                    new_mem[c] = self.reset_potential
                    spk_rec[-1].append(z)

                mem[l] = new_mem
                syn[l] = new_syn

                mem_rec[-1].append(mem[l])


        # return the final recorded membrane potential in the output layer, all membrane potentials,
        # and spikes
        return mem_rec[-1][-1], mem_rec, spk_rec