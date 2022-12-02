import torch
import torch.nn as nn

import numpy as np
from functools import partial

from model import SurrGradSpike
from rstdp import RSTDP

default_device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

class DSNN(nn.Module):
    def __init__(self,
        state_size,
        operation,
        weights_size,
        alpha,
        beta,
        is_spiking=True,
        threshold=.1,
        dtype=torch.float, 
        #device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device=default_device
        ):
        super(DSNN, self).__init__()

        self.dtype = dtype
        self.device = device

        self.state_size = state_size
        self.operation = operation

        self.weights = torch.zeros(weights_size, dtype=self.dtype, device=self.device, requires_grad=True)
        nn.init.normal_(self.weights, mean=0., std=.1)
        self.weights = nn.Parameter(self.weights)

        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

        self.spike_fn = SurrGradSpike.apply
        self.is_spiking = is_spiking


    def _get_initial_state(self, template, amount_state_vars):
        assert template.shape[1:] == self.state_size, \
            "Specified 'state_size' {} does not match size of tensor resulting from operation {}." \
                .format(self.state_size, template.shape[1:])

        return [torch.zeros_like(template, dtype=self.dtype, device=self.device) for _ in range(amount_state_vars)]


    def forward(self, inputs, state=(None, None)):
        mem, syn = state
        spk_out, mem_out, syn_out = [], [], []

        for t in range(inputs.shape[1]):
            h = self.operation(inputs[:, t, ...], self.weights)

            if mem is None or syn is None:
                mem, syn = self._get_initial_state(template=h, amount_state_vars=2)

            new_syn = self.alpha * syn + h
            new_mem = (self.beta * mem + new_syn)

            if self.is_spiking:
                mthr = new_mem - self.threshold
                out = self.spike_fn(mthr)
                c = (mthr > 0.)
                new_mem[c] = 0.
            else:
                out = torch.zeros_like(mem)
            
            spk_out.append(out)
            mem_out.append(new_mem)
            syn_out.append(new_syn)

            mem = new_mem
            syn = new_syn

        spk_out = torch.stack(spk_out, dim=1)
        mem_out = torch.stack(mem_out, dim=1)
        syn_out = torch.stack(syn_out, dim=1)

        return spk_out, (mem, syn), (mem_out, syn_out)


class DSRNN(DSNN):
    def __init__(self,
        state_size,
        operation,
        weights_size,
        alpha,
        beta,
        is_spiking=True,
        threshold=.1,
        dtype=torch.float, 
        device=torch.device('cpu')
    ):
        super(DSRNN, self).__init__(
            state_size, operation, weights_size, alpha, beta, is_spiking, threshold, dtype, device
        )

        num_units = np.prod(self.state_size)

        self.recurrent_weights = torch.zeros((num_units, num_units), dtype=self.dtype, device=self.device, requires_grad=True)
        nn.init.normal_(self.recurrent_weights, mean=0., std=.1)
        self.recurrent_weights = nn.Parameter(self.recurrent_weights)


    def forward(self, inputs, state=(None, None, None)):
        mem, syn, spk = state
        spk_rec, mem_rec, syn_rec = [], [], []

        for t in range(inputs.shape[1]):
            h = self.operation(inputs[:, t, ...], self.weights)

            if mem is None or syn is None or spk is None:
                mem, syn, spk = self._get_initial_state(template=h, amount_state_vars=3)

            # Calculate recurrent connections:
            # Convert spk tensor to size (batch_size, num_units)
            linear_spk = spk.view(spk.shape[0], -1)
            # Calculate recurrent inputs
            rec_h = torch.einsum('ab,bc->ac', linear_spk, self.recurrent_weights)
            # Reshape recurrent inputs back to size of spks (batch_size, state_size)
            rec_h = rec_h.view(spk.shape)

            new_syn = self.alpha * syn + h + rec_h
            new_mem = (self.beta * mem + new_syn)

            if self.is_spiking:
                mthr = new_mem - self.threshold
                out = self.spike_fn(mthr)
                c = (mthr > 0.)
                new_mem[c] = 0.
            else:
                out = torch.zeros_like(mem)
            
            spk_rec.append(out)
            mem_rec.append(new_mem)
            syn_rec.append(new_syn)

            mem = new_mem
            syn = new_syn
            spk = out

        spk_rec = torch.stack(spk_rec, dim=1)
        mem_rec = torch.stack(mem_rec, dim=1)
        syn_rec = torch.stack(syn_rec, dim=1)

        return spk_rec, (mem, syn, spk), (mem_rec, syn_rec)


def transform_state(state):
    state_ = []
    for i in state:
        if i > 0:
            state_.append(i)
            state_.append(0)
        else:
            state_.append(0)
            state_.append(abs(i))
    return torch.tensor(state_)


class RSTDPNet(nn.Module):
    def __init__(self, alpha, beta, threshold, architecture, simulation_time, weights, tau=16, tau_e=800,
                 A_plus=0.00001, A_minus=0.00001, C=0.01, device=default_device, dtype=torch.float):
        super(RSTDPNet, self).__init__()

        self.simulation_time = simulation_time
        self.device = device
        self.dtype = dtype

        self.l1 = DSNN(
            state_size=(architecture[1],),
            operation=partial(torch.einsum, 'ab,bc->ac'),
            weights_size=(architecture[0], architecture[1]),
            alpha=alpha,
            beta=beta,
            threshold=threshold,
            device=self.device,
            dtype=self.dtype
        )
        # Set 'requires_grad' to 'False', since value assignment is not supported otherwise
        self.l1.weights = nn.Parameter(weights[0][0].clone(), requires_grad=False)
        self.l2 = DSNN(
            state_size=(architecture[2],),
            operation=partial(torch.einsum, 'ab,bc->ac'),
            weights_size=(architecture[1], architecture[2]),
            alpha=alpha,
            beta=beta,
            threshold=threshold,
            device=self.device,
            dtype=self.dtype
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
            is_spiking=False,
            device=self.device,
            dtype=self.dtype
        )
        self.l3.weights = nn.Parameter(weights[0][2].clone(), requires_grad=False)

        self.rstdp=RSTDP(A_plus=A_plus, A_minus=A_minus, tau_plus=tau, tau_minus=tau, tau_e=tau_e, C=C,
            device=self.device, dtype=self.dtype)

    def forward(self, inputs, rstdp_state=(None, None, None)):
        # Two neuron encoding
        inputs = transform_state(inputs)
        # Expand analogue inputs for each timestep
        inputs = torch.tile(inputs, (1, self.simulation_time, 1)).to(self.device)

        # Calculate layers
        y_l1, _, _ = self.l1(inputs)
        y_l2, _, _ = self.l2(y_l1)
        _, mem_result, _ = self.l3(y_l2)

        # Calculate rstdp
        rstdp_out = self.rstdp(y_l1, y_l2, rstdp_state)
        
        return mem_result[0], rstdp_out
