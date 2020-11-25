"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(LSTM, self).__init__()

        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.embedding = nn.Embedding(num_classes, input_dim)

        self.Wgx = nn.Parameter(torch.normal(0.0, 1.0, size=(hidden_dim, input_dim)) * math.sqrt(2 / input_dim))
        self.Wgh = nn.Parameter(torch.normal(0.0, 1.0, size=(hidden_dim, hidden_dim)) * math.sqrt(2 / hidden_dim))
        self.bg = nn.Parameter(torch.zeros(hidden_dim))

        self.Wix = nn.Parameter(torch.normal(0.0, 1.0, size=(hidden_dim, input_dim)) * math.sqrt(2 / input_dim))
        self.Wih = nn.Parameter(torch.normal(0.0, 1.0, size=(hidden_dim, hidden_dim)) * math.sqrt(2 / hidden_dim))
        self.bi = nn.Parameter(torch.zeros(hidden_dim))

        self.Wfx = nn.Parameter(torch.normal(0.0, 1.0, size=(hidden_dim, input_dim)) * math.sqrt(2 / input_dim))
        self.Wfh = nn.Parameter(torch.normal(0.0, 1.0, size=(hidden_dim, hidden_dim))* math.sqrt(2 / hidden_dim))
        self.bf = nn.Parameter(torch.zeros(hidden_dim))

        self.Wox = nn.Parameter(torch.normal(0.0, 1.0, size=(hidden_dim, input_dim)) * math.sqrt(2 / input_dim))
        self.Woh = nn.Parameter(torch.normal(0.0, 1.0, size=(hidden_dim, hidden_dim)) * math.sqrt(2 / hidden_dim))
        self.bo = nn.Parameter(torch.zeros(hidden_dim))

        self.Wph = nn.Parameter(torch.normal(0.0, 1.0, size=(num_classes, hidden_dim)) * math.sqrt(2 / hidden_dim))
        self.bp = nn.Parameter(torch.zeros(num_classes))
        
    def forward(self, x):
        x = x.long() # Convert into appropriate format for input
        h_t = torch.zeros(self.hidden_dim)
        c_t = torch.zeros(self.hidden_dim)

        for seq_idx in range(self.seq_length):
            x_t = self.embedding(x[:, seq_idx, :])
            g_t = torch.tanh(x_t @ self.Wgx.T + h_t @ self.Wgh.T + self.bg)
            i_t = torch.sigmoid(x_t @ self.Wix.T + h_t @ self.Wih.T + self.bi)
            f_t = torch.sigmoid(x_t @ self.Wfx.T + h_t @ self.Wfh.T + self.bf)
            o_t = torch.sigmoid(x_t @ self.Wox.T + h_t @ self.Woh.T + self.bo)
            
            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t

        p_t = h_t @ self.Wph.T + self.bp
        out = torch.log(torch.softmax(p_t.reshape(x.shape[0], self.num_classes), 1))

        return out
