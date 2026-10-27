"""The student: one MLP from the unified observation (env.OBS) to joint targets for every behaviour.

u = tanh-free linear output; ctrl = clip(DOG + FULL * u, lo, hi) (env.Master.ctrl_of). Inputs are normalized with
fixed statistics computed from the training data (stored with the weights), like the experts' VecNormalize.
"""
import numpy as np
import torch
import torch.nn as nn

from env import OBS


class Net(nn.Module):
    def __init__(self, obs=OBS, hidden=(512, 512, 256), act=12):
        super().__init__()
        layers, n = [], obs
        for h in hidden:
            layers += [nn.Linear(n, h), nn.ELU()]
            n = h
        layers.append(nn.Linear(n, act))
        self.mlp = nn.Sequential(*layers)
        self.hidden = list(hidden)
        self.register_buffer("mean", torch.zeros(obs))
        self.register_buffer("std", torch.ones(obs))

    def forward(self, x):
        return self.mlp(torch.clamp((x - self.mean) / self.std, -10, 10))


class Student:
    """act(obs) -> joint targets (radians) for a Master env."""

    def __init__(self, path, master):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net = Net(hidden=ck["hidden"])
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.m = master

    def act(self, obs):
        with torch.no_grad():
            u = self.net(torch.as_tensor(obs, dtype=torch.float32)[None])[0].numpy()
        return self.m.ctrl_of(u)


def save(net, path, **extra):
    torch.save(dict(state=net.state_dict(), hidden=net.hidden, **extra), path)
