"""
This class defines the basic structure for a NoiseModel
object. A noise model object takes in a numpy matrix and
outputs another numpy matrix with the same shape.
"""
import numpy as np
import random
import copy
import pandas as pd
import string

chars = np.array(list(string.ascii_letters + string.digits)) # + string.punctuation


def generate_typo(cur_str):

  """
  Generates simple typos for categorical noise models

    -> Assumes Damerau-Levenshtein edit distance
       with all errors being within edit distance 2.
       It is a common assumption about typos.
  """

  str_lst = list(cur_str)

  # choose how many typo operations to carry out, usually between 1 and 2
  dt_chars_numb = np.random.randint(1,3)

  for ii in range(dt_chars_numb):

    if len(str_lst):
      # get random character index
      char_idx = np.random.choice(len(str_lst))

      if len(str_lst) > 1:
        # choose typo operation randomly
        typo_op = np.random.choice(4)

      else:
        # choose typo operation randomly
        typo_op = np.random.choice(3)

    else:
      char_idx = 0
      typo_op = 1 # only insertion is used as typo_op if str is ""

    if typo_op == 0: # and len(str_lst)
      # delete char
      del str_lst[char_idx]

    elif typo_op == 1:
      # insert char
      str_lst.insert(char_idx, np.random.choice(chars))

    elif typo_op == 2: # and len(str_lst)
      # replace char
      str_lst[char_idx] = np.random.choice(chars)

    elif typo_op == 3: # and len(str_lst)>1
      # permutation of chars
      tmp_char = str_lst[char_idx]
      str_lst[char_idx] = str_lst[(char_idx+1) % len(str_lst)]
      str_lst[(char_idx+1) % len(str_lst)] = tmp_char

  # if nothing in the cell, then change to missing as "?"
  if len(str_lst) == 0:
    str_lst = ["?"]

  return ''.join(str_lst)


class NoiseModel(object):

  """
  Creates a NoiseModel object.

  shape defines the shape of the input matrix N x p
  
  probability defines the probability that a 
  row in the input matrix is transformed

  feature_importance is a sorted sequence of 
  features of decreasing importance
  """
  def __init__(self,
               shape=(1,1), 
               probability=0,
               feature_importance=[],
               one_cell_flag=False):
    
    self.shape = shape
    self.probability = probability
    self.one_cell_flag = one_cell_flag

    if feature_importance == []:
      self.feature_importance = list(range(0,shape[1]))
    else:
      self.feature_importance = feature_importance

    """
    Argument error checks
    """

    #check to see if the shape provided is valid
    if len(shape) != 2 and len(shape) != 3:
      raise ValueError("Invalid shape: " + str(shape))

    if len(shape) == 2:
      if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("Invalid shape: " + str(shape))

    if len(shape) == 3:
      if shape[0] <= 0 or shape[1] <= 0 or shape[2] <= 0:
        raise ValueError("Invalid shape: " + str(shape))
          
    #check to see if the probability is valid
    if probability < 0 or probability > 1:
      raise ValueError("Invalid probability: " + str(probability))

    if sorted(self.feature_importance) != list(range(0,shape[1])):
      raise ValueError("Invalid feature_importance: " + str(self.feature_importance))

  """
  The apply function applies the noise model to some 
  subset of the data and performs the necessary error
  checks to make sure sizes are preserved.

  Accepts Numpy and Pandas DataFrame.
  """
  def apply(self, X):
    xshape = np.shape(X)

    if xshape != self.shape:
      raise ValueError("The input does not match the shape of the Noise Model")

    #sample from data
    N = self.shape[0]
    Ns = int(round(N*self.probability))
    all_indices = list(range(0,N))
    random.shuffle(all_indices)
    tocorrupt = all_indices[0:Ns]
    self.argselect = tocorrupt
    clean = all_indices[Ns:]

    # enforce previous order of tuples
    if not isinstance(X, pd.DataFrame):
      # Numpy ndarray implementation
      corrupt_data = np.empty(X.shape, dtype=object)
      corrupt_data[tocorrupt,:] = self.corrupt(X[tocorrupt,:])
      corrupt_data[clean,:] = X[clean,:]
    else:
      # Pandas DataFrame implementation
      corrupt_data = pd.DataFrame(index=X.index, columns=X.columns)
      corrupt_data.iloc[tocorrupt,:] = self.corrupt(X.iloc[tocorrupt,:])
      corrupt_data.iloc[clean,:] = X.iloc[clean,:]
    
    return corrupt_data, X

  """
  This method should be implemented by sub-classes
  """
  def corrupt(self, X):
    raise NotImplementedError("Please implement this method")

  """
  This method allows for dynamic reshaping of the noise model
  """
  def reshape(self, shape, feature_importance=[]):
    ret = copy.deepcopy(self)
    ret.shape = shape
    if feature_importance == []:
      ret.feature_importance = list(range(0,shape[1]))
    else:
      ret.feature_importance = feature_importance

    #check to see if the shape provided is valid
    if len(shape) != 2 and len(shape) != 3:
      raise ValueError("Invalid shape: " + str(shape))

    if len(shape) == 2:
      if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("Invalid shape: " + str(shape))

    if len(shape) == 3:
      if shape[0] <= 0 or shape[1] <= 0 or shape[2] <= 0:
        raise ValueError("Invalid shape: " + str(shape))

    if sorted(ret.feature_importance) != list(range(0,shape[1])):
      raise ValueError("Invalid feature_importance: " + str(ret.feature_importance))
      
    return ret






