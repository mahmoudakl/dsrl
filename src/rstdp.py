import torch
import numpy as np

class RSTDP(object):
    def __init__(self, 
        A_plus=1., 
        A_minus=1., 
        tau_plus=10., 
        tau_minus=10., 
        tau_e=10., 
        C=1.,
        time_step=1.,
        device='cpu',
        dtype=torch.float
    ):
        self._A_plus = A_plus
        self._A_minus = A_minus
        self._C = C

        self._exp_dec_plus = np.exp(-time_step/tau_plus)
        self._exp_dec_minus = np.exp(-time_step/tau_minus)
        self._e_trace_dec = tau_e / time_step

        self.device = device
        self.dtype = dtype

    def __call__(self, pre, post, state=(None, None, None)):
        # Get old state
        e, k_plus, k_minus = state

        # Set zero states, if input state is None (this can be used for resetting)
        if e is None:
            e = torch.zeros(pre.shape[0], pre.shape[-1], post.shape[-1], device=self.device, dtype=self.dtype)
        if k_plus is None:
            k_plus = torch.zeros(pre.shape[0], pre.shape[-1], post.shape[-1], device=self.device, dtype=self.dtype)
        if k_minus is None:
            k_minus = torch.zeros(pre.shape[0], pre.shape[-1], post.shape[-1], device=self.device, dtype=self.dtype)

        # Create lists for recording
        e_rec = []
        k_plus_rec = []
        k_minus_rec = []
        # Loop over each timestep in spikes
        for t in range(pre.shape[1]):
            # Calculate change in eligibility trace (e_dot)
            e_dot = -(e/self._e_trace_dec)
            e_dot += self._A_plus * k_plus * post[:, t, None, :] * self._C
            e_dot += -self._A_minus * k_minus * pre[:, t, ..., None] * self._C

            # Set new eligibility trace (e) from old e and change (e_dot)
            new_e = e + e_dot

            # Calculate new k-values
            new_k_plus = k_plus * self._exp_dec_plus + pre[:, t, ..., None]
            new_k_minus = k_minus * self._exp_dec_minus + post[:, t, None, :]

            # Append new values to recordings
            e_rec.append(new_e)
            k_plus_rec.append(new_k_plus)
            k_minus_rec.append(new_k_minus)
            # Set new values to variables
            e = new_e
            k_plus = new_k_plus
            k_minus = new_k_minus

        # Stack recordings into single array, along simulation_time dimension
        e_rec = torch.stack(e_rec, dim=1)
        k_plus_rec = torch.stack(k_plus_rec, dim=1)
        k_minus_rec = torch.stack(k_minus_rec, dim=1)

        # Return state and recordings
        return (e, k_plus, k_minus), (e_rec, k_plus_rec, k_minus_rec)
