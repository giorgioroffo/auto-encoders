# -*- coding: utf-8 -*-
"""

@author: Roffo
"""
import torch
import torch.nn as nn
import numpy as np


# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, nb_features, nb_neurons, ):
        super(SAE, self).__init__()
        # first layer: nb_neurons: number of hidden nodes 
        self.fc1 = nn.Linear(nb_features, nb_neurons)
        # second layer: nb_neurons input size, and nb_neurons/2 number of hidden nodes 
        self.fc2 = nn.Linear(nb_neurons, np.int(nb_neurons/2))
        # second layer: nb_neurons/2 input size, and nb_neurons number of hidden nodes
        self.fc3 = nn.Linear(np.int(nb_neurons/2), nb_neurons)
        #Output layes, size output is the same of size input
        self.fc4 = nn.Linear(nb_neurons, nb_features)
        # Activation function
        self.activation = nn.Sigmoid()
    # Forward propagation    
    def forward(self, x):
        # Activate the Encoded vector of the first FC   
        x = self.activation(self.fc1(x))
        # Activate the Encoded vector of the second FC   
        x = self.activation(self.fc2(x))
        # Activate the Encoded vector of the third FC   
        x = self.activation(self.fc3(x))
        # No activation function on the output layer to get the reconstructed vector
        x = self.fc4(x)
        return x