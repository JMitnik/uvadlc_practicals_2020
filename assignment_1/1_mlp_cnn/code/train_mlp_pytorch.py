"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
from torch.optim import Adam

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100


# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

def ensure_path(path_to_file):
    """Ensures the right directories exist for the creation of a file."""
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)


def plot_results(
    train_results, 
    test_results, 
    path_to_train_results = 'results/pytorch-mlp-train_loss.png', 
    path_to_test_results = 'results/pytorch-mlp-test_accs.png'
):
    """Plots the train(=acc) and test(=acc) charts for the results"""

    ensure_path(path_to_train_results)
    ensure_path(path_to_test_results)

    plt.plot(
        [data['iteration'] for data in train_results], 
        [data['loss'] for data in train_results],
    )
    plt.title('Loss across training set')
    plt.xlabel('steps')
    plt.ylabel('Losses')
    plt.savefig(path_to_train_results)
    plt.cla()

    plt.plot(
        [data['iteration'] for data in test_results], 
        [data['accuracy'] for data in test_results],
    )
    plt.title('Accuracy for the entire data-set, across steps, for Convnets')
    plt.xlabel('steps')
    plt.ylabel('accuracy')
    plt.savefig(path_to_test_results)
    


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """
    correct = (targets.argmax(1) == predictions.argmax(1)).sum().item()
    total = predictions.shape[0]

    return correct / total


def train():
    """
    Performs training and evaluation of MLP model.
  
    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    train_losses = []
    test_accs = []

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    path_to_data = FLAGS.data_dir #type: ignore
    validation_size = 0

    data_sets = cifar10_utils.read_data_sets(path_to_data, True, validation_size)
    
    train_dataset: DataSet = data_sets['train']
    test_dataset: DataSet = data_sets['test']
    in_size = train_dataset.images[0].flatten().shape[0]
    nr_labels = train_dataset.labels[0].shape[0]
    
    batch_size = FLAGS.batch_size #type: ignore
    nr_iterations = FLAGS.max_steps #type: ignore

    net = MLP(in_size, dnn_hidden_units, nr_labels).to(device)
    optimizer = Adam(params=net.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # neg_slope = FLAGS.neg_slope
    
    for iteration in range(nr_iterations):
        optimizer.zero_grad()
        X, y = train_dataset.next_batch(batch_size)
        X: torch.Tensor = torch.from_numpy(X).to(device)
        y: torch.Tensor = torch.from_numpy(y).to(device)
        X = X.flatten(1)
        preds = net(X).to(device)
        loss = loss_fn(preds, y.argmax(1))

        train_losses.append({
            'iteration': iteration,
            'loss': loss.item()
        })

        loss.backward()
        optimizer.step()

        if iteration % 50 == 0:
            print(f"Loss on iteration {iteration} is {loss.item()}")

        if iteration % FLAGS.eval_freq == 0: #type: ignore
            acc_scores = []
            test_dataset: DataSet = data_sets['test']
            latest_epochs_completed = test_dataset.epochs_completed
            target_epochs_completed = latest_epochs_completed + 1
            acc = 0
            
            with torch.no_grad():
                while test_dataset.epochs_completed < target_epochs_completed:
                    X_test, y_test = test_dataset.next_batch(batch_size)
                    X_test = torch.from_numpy(X_test).to(device)
                    X_test = X.flatten(1)
                    y_test = torch.from_numpy(y_test).to(device)
                    pred_test = net.forward(X_test)

                    acc = accuracy(pred_test, y_test)
                    acc_scores.append(acc)
            
            print(f"Average test accuracy on {iteration} is {acc}")
            acc = np.mean(acc_scores).item()
            test_accs.append(({
                'iteration': iteration,
                'accuracy': acc
            }))
    
    plot_results(train_losses, test_accs)

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
