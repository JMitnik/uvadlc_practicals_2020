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
from importlib.resources import path

import os
from platform import python_branch
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import TextDataset
from model import TextGenerationModel

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import utils
from utils import ResultsWriter

###############################################################################


def train(config):
    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size)

    # Initialize the model that we are going to use
    model = TextGenerationModel(
        config.batch_size,
        config.seq_length,
        dataset.vocab_size,
        config.lstm_num_hidden,
        config.lstm_num_layers,
        device=config.device
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    res_writer = utils.ResultsWriter(
        config.summary_path,
        f'{config.label}--nr_layers-{config.lstm_num_layers}--lr-{config.learning_rate}--nr_hidden-{config.lstm_num_hidden}--seqs-{config.seq_length}--{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}',
        {
            'label': config.label,
            'nr_to_sample': config.nr_to_sample,
            'temperature': config.temperature,
            'txt_file': config.txt_file,
            'lr': config.learning_rate,
            'seq_length': config.seq_length,
            'lstm_hidden': config.lstm_num_hidden,
            'lstm_num_layers': config.lstm_num_layers
        }
    )

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        X = batch_inputs
        y = batch_targets
        X = torch.stack(X, 1).to(config.device)
        y = torch.stack(y, 1).to(config.device)
        preds, _ = model(X)

        # Flatten preds and y
        preds = preds.reshape(-1, preds.shape[2])
        y = y.reshape(-1)

        loss = F.cross_entropy(preds, y)
        loss.backward()
        optimizer.step()


        optimizer.zero_grad()
        accuracy = (preds.argmax(1) == y).sum().item() / len(y)


        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if (step + 1) % config.save_every == 0:
            if len(res_writer.losses) == 0 or loss < res_writer.losses[-1]:
                res_writer.save_model(model)

        if (step + 1) % config.print_every == 0:
            res_writer.add_accuracy(accuracy, step)
            res_writer.add_loss(loss, step)

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                    Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
                    ))

        if (step + 1) % config.sample_every == 0:
            start_word, start_idx = dataset.sample_random_chars(1)
            current_sent = start_idx
            
            with torch.no_grad():
                h_current = None
                for nr in range(config.nr_to_sample - 1):
                    X = torch.tensor(current_sent[-1]).to(config.device).reshape(1, 1)
                    pred, h_current = model(X, h_current)
                    pred = pred.squeeze()
                    sampled_token = utils.sample(pred, config.temperature)
                    current_sent.append(sampled_token)
                
            decoded_sent = dataset.convert_to_string(current_sent)

            # Add character
            res_writer.add_sampled_text(start_word[0], decoded_sent, step)

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error,
            # check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break
        
    res_writer.summarize_training()
    res_writer.stop()
    print('Done training.')


###############################################################################
###############################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True,
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1e6),
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100,
                        help='How often to sample from the model')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='How often to save model')
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")

    # If needed/wanted, feel free to add more arguments
    parser.add_argument('--label', type=str, default='', help='Readable label as prefix')
    parser.add_argument('--nr_to_sample', type=int, default=30, help='Number of samples to generate')
    parser.add_argument('--temperature', type=float, default=1, help='Temperature to add to the sampling')

    config = parser.parse_args()

    # Train the model
    train(config)
