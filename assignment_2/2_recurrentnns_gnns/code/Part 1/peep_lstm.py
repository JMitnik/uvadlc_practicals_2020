"""
This module implements a LSTM with peephole connections in PyTorch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class peepLSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(peepLSTM, self).__init__()

        self.device = device
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.embedding = nn.Embedding(num_classes, input_dim)

        self.Wcx = nn.Parameter(torch.normal(0.0, 1.0, size=(hidden_dim, input_dim)).to(device) * math.sqrt(2 / input_dim)).to(device)
        self.bc = nn.Parameter(torch.zeros(hidden_dim)).to(device)

        self.Wix = nn.Parameter(torch.normal(0.0, 1.0, size=(hidden_dim, input_dim)).to(device) * math.sqrt(2 / input_dim)).to(device)
        self.Wih = nn.Parameter(torch.normal(0.0, 1.0, size=(hidden_dim, hidden_dim)).to(device) * math.sqrt(2 / hidden_dim)).to(device)
        self.bi = nn.Parameter(torch.zeros(hidden_dim)).to(device)

        self.Wfx = nn.Parameter(torch.normal(0.0, 1.0, size=(hidden_dim, input_dim)).to(device) * math.sqrt(2 / input_dim)).to(device)
        self.Wfh = nn.Parameter(torch.normal(0.0, 1.0, size=(hidden_dim, hidden_dim)).to(device)* math.sqrt(2 / hidden_dim)).to(device)
        self.bf = nn.Parameter(torch.zeros(hidden_dim)).to(device)

        self.Wox = nn.Parameter(torch.normal(0.0, 1.0, size=(hidden_dim, input_dim)).to(device) * math.sqrt(2 / input_dim)).to(device)
        self.Woh = nn.Parameter(torch.normal(0.0, 1.0, size=(hidden_dim, hidden_dim)).to(device) * math.sqrt(2 / hidden_dim)).to(device)
        self.bo = nn.Parameter(torch.zeros(hidden_dim)).to(device)

        self.Wph = nn.Parameter(torch.normal(0.0, 1.0, size=(num_classes, hidden_dim)).to(device) * math.sqrt(2 / hidden_dim)).to(device)
        self.bp = nn.Parameter(torch.zeros(num_classes)).to(device)
        

    def forward(self, x):
        x = x.to(self.device)
        x = x.long().to(self.device) # Convert into appropriate format for input
        h_t = torch.zeros(self.hidden_dim).to(self.device)
        c_t = torch.zeros(self.hidden_dim).to(self.device)

        for seq_idx in range(self.seq_length):
            x_t = self.embedding(x[:, seq_idx, :]).to(self.device)

            f_t = torch.sigmoid(x_t @ self.Wfx.T + c_t @ self.Wfh.T + self.bf)
            i_t = torch.sigmoid(x_t @ self.Wix.T + c_t @ self.Wih.T + self.bi)
            o_t = torch.sigmoid(x_t @ self.Wox.T + c_t @ self.Woh.T + self.bo)
            
            c_t = (torch.sigmoid(x_t @ self.Wcx.T + self.bc) * i_t) + c_t * f_t
            h_t = torch.tanh(c_t) * o_t

        p_t = h_t @ self.Wph.T + self.bp
        out = torch.log(torch.softmax(p_t.reshape(x.shape[0], self.num_classes), 1))

        return out
