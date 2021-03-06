from logging import root
from datetime import datetime
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

import torch
import os
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np
import pandas as pd

def ensure_path(path_to_file):
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)

    return path_to_file

class ResultsWriter:
    def __init__(
        self, 
        root_path_to_results,
        base_label, 
        exp_name,
        exp_params
    ) -> None:
        self.start_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        self.base_label = base_label
        self.exp_name = exp_name
        self.root_path_to_results = root_path_to_results
        
        self.path_to_results = f'{root_path_to_results}/{exp_name}'

        ensure_path(f'{self.path_to_results}/test.csv')

        self._init_yaml(f'{self.path_to_results}/experiment.yaml', {
            'exp': exp_name,
            'start_time': self.start_time,
            **exp_params
        })

        self.sw_writer = SummaryWriter(
            log_dir=self.path_to_results,
        )

        self.path_to_text_file = f'{self.path_to_results}/sampled_text.txt'
        self._init_sampled_text(self.path_to_text_file)

        self.losses = []
        self.accs = []

    def _init_sampled_text(self, path_to_text_file):
        with open (path_to_text_file, 'w') as file:
            file.write("Start of sampled-text\n\n")
    
    def _init_yaml(self, path_to_yaml, params):
        with open(path_to_yaml, 'w') as f:
            yaml.dump([params], f)
    
    def add_loss(self, loss, iter):
        self.losses.append(loss)
        self.sw_writer.add_scalar('Loss', loss, iter)
    
    def add_accuracy(self, acc, iter):
        self.accs.append(acc)
        self.sw_writer.add_scalar('Acc', acc, iter)

    def add_sampled_text(self, start, text, step):
        with open (self.path_to_text_file, 'a') as file:
            file.write(
                f"On step {step} we generate the following, given the start '{start}' :\n"
                f"=====\n"
                f"{text}"
                f"\n \n"
            )

    def save_model(self, model: nn.Module):
        torch.save(model.state_dict(), f'{self.path_to_results}/{model._get_name()}.pt')

    def stop(self):
        self.sw_writer.close()

    def summarize_training(self):
        # Save final results
        df = pd.DataFrame({ 'loss': self.losses, 'accs': self.accs })
        df.to_csv(f'{self.path_to_results}/results.csv')

        create_loss_acc_plots(
            self.losses,
            self.accs,
            self.base_label,
            f"{self.path_to_results}/training-plots"
        )

def sample(logits, temperature = 1, use_temperature = 1):
    logits = logits.detach()
    probs: torch.Tensor = torch.softmax(logits * temperature, 0)

    # If temperature is 1, we will do 
    if use_temperature:
        return torch.multinomial(probs, 1).item()

    # Else, for now we return greedy
    return torch.argmax(probs, 0).item()

def create_loss_acc_plots(losses, accs, title, base_filename):
    plt.plot(losses, color='r')
    plt.title(f"Loss run:{title}")
    plt.savefig(f"{base_filename}-loss.png")
    plt.cla()
    
    plt.plot(accs, color='g')
    plt.title(f"Accs run:{title}")
    plt.savefig(f"{base_filename}-accs.png")
    plt.cla()