# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        self.embedding_size = int(lstm_num_hidden / 2)

        self.embedding = nn.Embedding(vocabulary_size, self.embedding_size)
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layer = lstm_num_layers
        self.device = device
        
        self.lstm = nn.LSTM(
            input_size=self.embedding_size, 
            hidden_size=lstm_num_hidden, 
            num_layers=lstm_num_layers,
            batch_first=True
        ).to(device)

        self.hid_2out = nn.Linear(lstm_num_hidden, vocabulary_size)

    def forward(self, x, h_current = None, teacher_forcing=False):
        x = self.embedding(x)

        if h_current:
            out, h = self.lstm(x, h_current)
        else:
            out, h = self.lstm(x)
        
        out = self.hid_2out(out)

        return out, h
