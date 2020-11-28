from logging import root
from datetime import datetime
import torch.nn as nn
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import yaml

def ensure_path(path_to_file):
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)

    return path_to_file

class ResultsWriter:
    def __init__(
        self, 
        root_path_to_results, 
        exp_name,
        exp_params
    ) -> None:
        self.start_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
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

        self.losses = []
        self.accs = []
    
    def _init_yaml(self, path_to_yaml, params):
        with open(path_to_yaml, 'w') as f:
            yaml.dump([params], f)
    
    def add_loss(self, loss, iter):
        self.losses.append(loss)
        self.sw_writer.add_scalar('Loss', loss, iter)
    
    def add_accuracy(self, acc, iter):
        self.accs.append(acc)
        self.sw_writer.add_scalar('Acc', acc, iter)

    def save_model(self, model: nn.Module):
        torch.save(model.state_dict(), f'{self.path_to_results}/{model.__get_name()}.pt')

    def stop(self):
        self.sw_writer.close()