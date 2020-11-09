"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from matplotlib.pyplot import xlabel
import numpy as np
import os

from numpy.lib import utils
from cifar10_utils import DataSet
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def ensure_path(path_to_file):
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)

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
    """
    correct = (targets.argmax(1) == predictions.argmax(1)).sum()
    total = predictions.shape[0]

    return correct / total


def train():
    """
    Performs training and evaluation of MLP model.
    """
    train_losses = []
    test_accs = []

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42) # type: ignore

    lr = FLAGS.learning_rate

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units: #type: ignore
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",") #type: ignore
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    path_to_data = FLAGS.data_dir #type: ignore
     # TODO: Maybe can be better?
    validation_size = 2000
    data_sets = cifar10_utils.read_data_sets(path_to_data, True, validation_size)

    batch_size = FLAGS.batch_size #type: ignore
    nr_iterations = FLAGS.max_steps #type: ignore

    train_dataset: DataSet = data_sets['train']
    test_dataset: DataSet = data_sets['test']
    in_size = train_dataset.images[0].flatten().shape[0]
    nr_labels = train_dataset.labels[0].shape[0]

    mlp = MLP(in_size, dnn_hidden_units, nr_labels)
    loss_module = CrossEntropyModule()
    
    # Training loop
    for iteration in range(nr_iterations):
        X, y = train_dataset.next_batch(batch_size)
        # Reshape to correspond to single-vector
        X = X.reshape(batch_size, -1)
        
        preds = mlp.forward(X)
        loss = loss_module.forward(preds, y)
        grad_loss = loss_module.backward(preds, y)

        mlp.backward(grad_loss)

        # Perform gradient-descent
        mlp.sgd(lr)
        train_losses.append({
            'iteration': iteration,
            'loss': loss.item()
        })

        if iteration % FLAGS.eval_freq == 0: #type: ignore
            X_test = test_dataset.images
            X_test = X_test.reshape(X_test.shape[0], -1)
            y_test = test_dataset.labels
            pred = mlp.forward(X_test)
            acc = accuracy(pred, y_test)

            test_accs.append(({
                'iteration': iteration,
                'accuracy': acc
            }))
    
    ensure_path('results/train_loss.png')
    ensure_path('results/test_accs.png')

    plt.plot(
        [data['iteration'] for data in train_losses], 
        [data['loss'] for data in train_losses],
    )
    plt.title('Loss across training set')
    plt.xlabel('steps')
    plt.ylabel('Losses')
    plt.savefig('results/train_loss.png')
    plt.cla()

    plt.plot(
        [data['iteration'] for data in test_accs], 
        [data['accuracy'] for data in test_accs],
    )
    plt.title('Accuracy for the entire data-set, across steps')
    plt.xlabel('steps')
    plt.ylabel('accuracy')
    plt.savefig('results/test_accs.png')
    

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
