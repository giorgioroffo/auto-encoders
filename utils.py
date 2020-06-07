# -*- coding: utf-8 -*-
"""
UTILS 

@author: Roffo
"""
import numpy as np
import torch

# Converting the data into an array with users in lines and movies in columns
def convert(data, nb_users, nb_movies):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
        
    return new_data

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
def torch_like_notLike_encoding(training_set, test_set):
      
      # Converting the data into Torch tensors
      training_set = torch.FloatTensor(training_set)
      test_set = torch.FloatTensor(test_set)

      training_set[training_set == 0] = -1
      training_set[training_set == 1] = 0
      training_set[training_set == 2] = 0
      training_set[training_set >= 3] = 1
      test_set[test_set == 0] = -1
      test_set[test_set == 1] = 0
      test_set[test_set == 2] = 0
      test_set[test_set >= 3] = 1
      
      return training_set, test_set