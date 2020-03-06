"""
This module defines a collection of random
noise modules--that is they do not use the
features.
"""
import numpy
from . import NoiseModel
import pandas
from scipy.stats import bernoulli


def bound_number(value, low, high):
  return max(low, min(high, value))

vbound = numpy.vectorize(bound_number)


"""
This model implements Gaussian Noise
"""
class GaussianNoiseModel(NoiseModel.NoiseModel):

  """
  Mu and Sigma are Params
  """
  def __init__(self,
               shape,
               probability=0,
               feature_importance=[],
               one_cell_flag=False,
               mu=0,
               sigma=1,
               scale=numpy.array([]),
               int_cast=numpy.array([]),
               p_cell=-1,
               mu_scaling=True):

    super(GaussianNoiseModel, self).__init__(shape,
                                             probability,
                                             feature_importance,
                                             one_cell_flag)
    self.mu = mu
    self.mu_scaling = mu_scaling
    self.sigma = sigma
    self.scale = scale # numpy array
    self.p_cell = p_cell

    # cast the noise quantity to int, if True
    if not int_cast.size:
      self.int_cast = numpy.zeros(shape[1], dtype=bool)
    else:
      self.int_cast = int_cast

  def corrupt(self, X):

    Ns = numpy.shape(X)[0]
    ps = numpy.shape(X)[1]

    if not self.one_cell_flag:

      if not self.scale.size:
        scale = 1.0
        mu_scale = 1.0
      else:
        scale = self.scale
        mu_scale = scale if self.mu_scaling else 1.0

      Z = numpy.random.randn(Ns,ps)*self.sigma*scale + self.mu*mu_scale

      if self.p_cell <=0:
        mask_mtx = numpy.ones((Ns, ps))

      else:
        mask_mtx = numpy.random.uniform(0.0, 1.0, (Ns, ps)) <= self.p_cell

      Z = Z * mask_mtx

      return vbound(X + Z, numpy.finfo(float).min, numpy.finfo(float).max)

    else:
      Y = numpy.copy(X)
      for i in range(0, Ns):
        a = numpy.random.choice(ps)

        if not self.scale.size:
          scale = 1.0
          mu_scale = 1.0
        else:
          scale = self.scale[a]
          mu_scale = scale if self.mu_scaling else 1.0

        if self.int_cast[a]:
          noiz = int(numpy.ceil(numpy.random.randn()*self.sigma*scale + self.mu*mu_scale))
          Y[i,a] = bound_number(Y[i,a] + noiz, numpy.iinfo(int).min, numpy.iinfo(int).max)
        else:
          noiz = numpy.random.randn()*self.sigma*scale + self.mu*mu_scale
          Y[i,a] = bound_number(Y[i,a] + noiz, numpy.finfo(float).min, numpy.finfo(float).max)

      return Y

  def corrupt_elem(self, Y, idxs, idx_map_num):

    for idx_0, idx_1 in idxs:

      if not self.scale.size:
        scale = 1.0
        mu_scale = 1.0
      else:
        scale = self.scale[idx_map_num[idx_1]]
        mu_scale = scale if self.mu_scaling else 1.0

      if self.int_cast[idx_map_num[idx_1]]:
        noiz = int(numpy.ceil(numpy.random.randn()*self.sigma*scale + self.mu*mu_scale))
        Y[idx_0, idx_1] = bound_number(Y[idx_0, idx_1] + noiz, numpy.iinfo(int).min, numpy.iinfo(int).max)
      else:
        noiz = numpy.random.randn()*self.sigma*scale + self.mu*mu_scale
        Y[idx_0, idx_1] = bound_number(Y[idx_0, idx_1] + noiz, numpy.finfo(float).min, numpy.finfo(float).max)



"""
This model implements Laplace Noise
"""
class LaplaceNoiseModel(NoiseModel.NoiseModel):

  """
  mu and b are params
  """
  def __init__(self, 
               shape, 
               probability=0,
               feature_importance=[],
               one_cell_flag=False,
               mu=0,
               b=1,
               scale=numpy.array([]),
               int_cast=numpy.array([]),
               p_cell=-1,
               mu_scaling=True):

    super(LaplaceNoiseModel, self).__init__(shape, 
                                            probability, 
                                            feature_importance,
                                            one_cell_flag)
    self.mu = mu
    self.mu_scaling = mu_scaling
    self.b = b
    self.scale = scale # numpy array
    self.p_cell = p_cell

    # cast the noise quantity to int, if True
    if not int_cast.size:
      self.int_cast = numpy.zeros(shape[1], dtype=bool)
    else:
      self.int_cast = int_cast

  def corrupt(self, X):

    Ns = numpy.shape(X)[0]
    ps = numpy.shape(X)[1]

    if not self.one_cell_flag:

      if not self.scale.size:
        scale = 1.0
        mu_scale = 1.0
      else:
        scale = self.scale
        mu_scale = scale if self.mu_scaling else 1.0

      Z = numpy.random.laplace(self.mu*mu_scale, self.b*scale, (Ns,ps))

      if self.p_cell <=0:
        mask_mtx = numpy.ones((Ns, ps))

      else:
        mask_mtx = numpy.random.uniform(0.0, 1.0, (Ns, ps)) <= self.p_cell

      Z = Z * mask_mtx

      return vbound(X + Z, numpy.finfo(float).min, numpy.finfo(float).max)

    else:
      Y = numpy.copy(X)
      for i in range(0, Ns):
        a = numpy.random.choice(ps)

        if not self.scale.size:
          scale = 1.0
          mu_scale = 1.0
        else:
          scale = self.scale[a]
          mu_scale = scale if self.mu_scaling else 1.0

        if self.int_cast[a]:
          noiz = int(numpy.ceil(numpy.random.laplace(self.mu*mu_scale, self.b*scale)))
          Y[i,a] = bound_number(Y[i,a] + noiz, numpy.iinfo(int).min, numpy.iinfo(int).max)
        else:
          noiz = numpy.random.laplace(self.mu*mu_scale, self.b*scale)
          Y[i,a] = bound_number(Y[i,a] + noiz, numpy.finfo(float).min, numpy.finfo(float).max)

      return Y

  def corrupt_elem(self, Y, idxs, idx_map_num):

    for idx_0, idx_1 in idxs:

      if not self.scale.size:
        scale = 1.0
        mu_scale = 1.0
      else:
        scale = self.scale[idx_map_num[idx_1]]
        mu_scale = scale if self.mu_scaling else 1.0

      if self.int_cast[idx_map_num[idx_1]]:
        noiz = int(numpy.ceil(numpy.random.laplace(self.mu*mu_scale, self.b*scale)))
        Y[idx_0, idx_1] = bound_number(Y[idx_0, idx_1] + noiz, numpy.iinfo(int).min, numpy.iinfo(int).max)
      else:
        noiz = numpy.random.laplace(self.mu*mu_scale, self.b*scale)
        Y[idx_0, idx_1] = bound_number(Y[idx_0, idx_1] + noiz, numpy.finfo(float).min, numpy.finfo(float).max)



"""
This model implements Log Normal Noise
"""
class LogNormalNoiseModel(NoiseModel.NoiseModel):

  """
  mu and sigma are params
  """
  def __init__(self,
               shape,
               probability=0,
               feature_importance=[],
               one_cell_flag=False,
               mu=0,
               sigma=1,
               scale=numpy.array([]),
               int_cast=numpy.array([]),
               p_cell=-1):

    super(LogNormalNoiseModel, self).__init__(shape,
                                              probability,
                                              feature_importance,
                                              one_cell_flag)
    self.mu = mu
    self.sigma = sigma
    self.scale = scale # numpy array
    self.p_cell = p_cell

    # cast the noise quantity to int, if True
    if not int_cast.size:
      self.int_cast = numpy.zeros(shape[1], dtype=bool)
    else:
      self.int_cast = int_cast

  def corrupt(self, X):

    Ns = numpy.shape(X)[0]
    ps = numpy.shape(X)[1]

    if not self.one_cell_flag:

      if not self.scale.size:
        scale = 1.0
      else:
        scale = self.scale

      Z = numpy.random.lognormal(self.mu, self.sigma, (Ns,ps))*scale

      if self.p_cell <=0:
        mask_mtx = numpy.ones((Ns, ps))

      else:
        mask_mtx = numpy.random.uniform(0.0, 1.0, (Ns, ps)) <= self.p_cell

      Z = Z * mask_mtx

      return vbound(X + Z, numpy.finfo(float).min, numpy.finfo(float).max)

    else:
      Y = numpy.copy(X)
      for i in range(0, Ns):
        a = numpy.random.choice(ps)

        if not self.scale.size:
          scale = 1.0
        else:
          scale = self.scale[a]

        if self.int_cast[a]:
          noiz = int(numpy.ceil(numpy.random.lognormal(self.mu, self.sigma)*scale))
          Y[i,a] = bound_number(Y[i,a] + noiz, numpy.iinfo(int).min, numpy.iinfo(int).max)
        else:
          noiz = numpy.random.lognormal(self.mu, self.sigma)*scale
          Y[i,a] = bound_number(Y[i,a] + noiz, numpy.finfo(float).min, numpy.finfo(float).max)

      return Y

  def corrupt_elem(self, Y, idxs, idx_map_num):

    for idx_0, idx_1 in idxs:

      if not self.scale.size:
        scale = 1.0
      else:
        scale = self.scale[idx_map_num[idx_1]]

      if self.int_cast[idx_map_num[idx_1]]:
        noiz = int(numpy.ceil(numpy.random.lognormal(self.mu, self.sigma)*scale))
        Y[idx_0, idx_1] = bound_number(Y[idx_0, idx_1] + noiz, numpy.iinfo(int).min, numpy.iinfo(int).max)
      else:
        noiz = numpy.random.lognormal(self.mu, self.sigma)*scale
        Y[idx_0, idx_1] = bound_number(Y[idx_0, idx_1] + noiz, numpy.finfo(float).min, numpy.finfo(float).max)


"""
This model implements Log Normal Noise
"""
class Mixture2GaussiansNoiseModel(NoiseModel.NoiseModel):

  """
  mu and sigma are params
  """
  def __init__(self,
               shape,
               probability=0,
               feature_importance=[],
               one_cell_flag=False,
               pc_1=0.5,
               mu_1=0.,
               sigma_1=1.,
               mu_2=0.,
               sigma_2=1.,
               scale=numpy.array([]),
               int_cast=numpy.array([]),
               p_cell=-1,
               mu_scaling=True):

    super(Mixture2GaussiansNoiseModel, self).__init__(shape,
                                                      probability,
                                                      feature_importance,
                                                      one_cell_flag)
    self.pc_1 = pc_1
    self.mu_1 = mu_1
    self.sigma_1 = sigma_1
    self.mu_2 = mu_2
    self.sigma_2 = sigma_2
    self.scale = scale # numpy array
    self.p_cell = p_cell
    self.mu_scaling = mu_scaling

    # cast the noise quantity to int, if True
    if not int_cast.size:
      self.int_cast = numpy.zeros(shape[1], dtype=bool)
    else:
      self.int_cast = int_cast

  def corrupt(self, X):

    Ns = numpy.shape(X)[0]
    ps = numpy.shape(X)[1]

    if not self.one_cell_flag:

      if not self.scale.size:
        scale = 1.0
        mu_scale = 1.0
      else:
        scale = self.scale
        mu_scale = scale if self.mu_scaling else 1.0

      pick_mtx = bernoulli.rvs(self.pc_1, size=(Ns,ps))
      Z_1 = numpy.random.randn(Ns,ps)*self.sigma_1*scale + self.mu_1*mu_scale
      Z_2 = numpy.random.randn(Ns,ps)*self.sigma_2*scale + self.mu_2*mu_scale
      Z = pick_mtx*Z_1 + (1-pick_mtx)*Z_2

      if self.p_cell <=0:
        mask_mtx = numpy.ones((Ns, ps))

      else:
        mask_mtx = numpy.random.uniform(0.0, 1.0, (Ns, ps)) <= self.p_cell

      Z = Z * mask_mtx

      return vbound(X + Z, numpy.finfo(float).min, numpy.finfo(float).max)

    else:
      Y = numpy.copy(X)
      for i in range(0, Ns):
        a = numpy.random.choice(ps)

        if not self.scale.size:
          scale = 1.0
          mu_scale = 1.0
        else:
          scale = self.scale[a]
          mu_scale = scale if self.mu_scaling else 1.0

        pick_elem = bernoulli.rvs(self.pc_1)
        z_1 = numpy.random.randn()*self.sigma_1*scale + self.mu_1*mu_scale
        z_2 = numpy.random.randn()*self.sigma_2*scale + self.mu_2*mu_scale

        if self.int_cast[a]:
          noiz = int(numpy.ceil(pick_elem*z_1 + (1-pick_elem)*z_2))
          Y[i,a] = bound_number(Y[i,a] + noiz, numpy.iinfo(int).min, numpy.iinfo(int).max)
        else:
          noiz = pick_elem*z_1 + (1-pick_elem)*z_2
          Y[i,a] = bound_number(Y[i,a] + noiz, numpy.finfo(float).min, numpy.finfo(float).max)

      return Y

  def corrupt_elem(self, Y, idxs, idx_map_num):

    for idx_0, idx_1 in idxs:

      if not self.scale.size:
        scale = 1.0
        mu_scale = 1.0
      else:
        scale = self.scale[idx_map_num[idx_1]]
        mu_scale = scale if self.mu_scaling else 1.0

      pick_elem = bernoulli.rvs(self.pc_1)
      z_1 = numpy.random.randn()*self.sigma_1*scale + self.mu_1*mu_scale
      z_2 = numpy.random.randn()*self.sigma_2*scale + self.mu_2*mu_scale

      if self.int_cast[idx_map_num[idx_1]]:
        noiz = int(numpy.ceil(pick_elem*z_1 + (1-pick_elem)*z_2))
        Y[idx_0, idx_1] = bound_number(Y[idx_0, idx_1] + noiz, numpy.iinfo(int).min, numpy.iinfo(int).max)
      else:
        noiz = pick_elem*z_1 + (1-pick_elem)*z_2
        Y[idx_0, idx_1] = bound_number(Y[idx_0, idx_1] + noiz, numpy.finfo(float).min, numpy.finfo(float).max)


"""
Zipfian Noise, simulates high-magnitude outliers
"""
class ZipfNoiseModel(NoiseModel.NoiseModel):

  """
  z is the Zipfian Scale Parameter
  """
  def __init__(self,
               shape,
               probability=0,
               feature_importance=[],
               one_cell_flag=False,
               z=3,
               scale=numpy.array([]),
               int_cast=numpy.array([]),
               active_neg=False,
               p_cell=-1):

    super(ZipfNoiseModel, self).__init__(shape,
                                         probability,
                                         feature_importance,
                                         one_cell_flag)

    self.z = z
    self.scale = scale # numpy array
    self.p_cell = p_cell

    # cast the noise quantity to int, if True
    if not int_cast.size:
      self.int_cast = numpy.zeros(shape[1], dtype=bool)
    else:
      self.int_cast = int_cast

    # flag to generate negative values also
    # zipf_val*numpy.random.choice([-1,1])
    self.active_neg = active_neg

  def corrupt(self, X):

    Ns = numpy.shape(X)[0]
    ps = numpy.shape(X)[1]

    if not self.one_cell_flag:

      if not self.scale.size:
        scale = 1.0
      else:
        scale = self.scale

      Z = numpy.random.zipf(self.z, (Ns,ps))*scale

      if self.p_cell <=0:
        mask_mtx = numpy.ones((Ns, ps))

      else:
        mask_mtx = numpy.random.uniform(0.0, 1.0, (Ns, ps)) <= self.p_cell

      Z = Z * mask_mtx

      return vbound(X + Z, numpy.finfo(float).min, numpy.finfo(float).max)

    else:
      Y = numpy.copy(X)
      for i in range(0, Ns):
        a = numpy.random.choice(ps)

        if not self.scale.size:
          scale = 1.0
        else:
          scale = self.scale[a]

        if self.active_neg:
          sign_val = numpy.random.choice([-1,1])
        else:
          sign_val = 1

        if self.int_cast[a]:
          noiz = int(numpy.ceil(numpy.random.zipf(self.z)*scale))
          Y[i,a] = bound_number(Y[i,a] + noiz*sign_val, numpy.iinfo(int).min, numpy.iinfo(int).max)
        else:
          noiz = numpy.random.zipf(self.z)*scale
          Y[i,a] = bound_number(Y[i,a] + noiz*sign_val, numpy.finfo(float).min, numpy.finfo(float).max)

      return Y

  def corrupt_elem(self, Y, idxs, idx_map_num):

    for idx_0, idx_1 in idxs:

      if not self.scale.size:
        scale = 1.0
      else:
        scale = self.scale[idx_map_num[idx_1]]

      if self.active_neg:
        sign_val = numpy.random.choice([-1,1])
      else:
        sign_val = 1

      if self.int_cast[idx_map_num[idx_1]]:
        noiz = int(numpy.ceil(numpy.random.zipf(self.z)*scale))
        Y[idx_0, idx_1] = bound_number(Y[idx_0, idx_1] + noiz*sign_val, numpy.iinfo(int).min, numpy.iinfo(int).max)

      else:
        noiz = numpy.random.zipf(self.z)*scale
        Y[idx_0, idx_1] = bound_number(Y[idx_0, idx_1] + noiz*sign_val, numpy.finfo(float).min, numpy.finfo(float).max)


"""
Simulates Random Errors for Categorical Features
Inserts both Typos and Category Changes
"""
class CategoricalNoiseModel(NoiseModel.NoiseModel):

  """
  The order in 'cats_name_lists' should be the same as in
  'cats_probs_list', such that for each column/feature their
  category names and probabilities match
  """
  
  def __init__(self,
               shape,
               cats_name_lists,
               probability=0,
               feature_importance=[],
               cats_probs_list=[], 
               typo_prob=0.01,
               alpha_prob=1.0,
               p_cell=0.01,
               one_cell_flag=False):

    super(CategoricalNoiseModel, self).__init__(shape,
                                                probability, 
                                                feature_importance,
                                                one_cell_flag)

    self.cats_name_lists = cats_name_lists
    self.cats_probs_list = cats_probs_list
    self.typo_prob = typo_prob
    self.alpha_prob = alpha_prob
    self.p_cell = p_cell

    if not self.cats_probs_list:
      # if cats_probs_list not provided assume that each feature/column has 
      #   uniform distribution on its categories
      self.cats_probs_list = [numpy.ones(len(self.cats_name_lists[i])) / float(len(self.cats_name_lists[i])) 
                              for i in range(len(self.cats_name_lists))]

    self.cats_probs_list = [self.cats_probs_list[i]**alpha_prob / numpy.sum(self.cats_probs_list[i]**alpha_prob)
                            for i in range(len(self.cats_name_lists))]

    # renormalize categorical probabilities for each feature
    #   as to include typo probability
    self.cats_probs_list = [numpy.append(cats_prob*(1-typo_prob), typo_prob) 
                            for cats_prob in self.cats_probs_list]

  def corrupt(self, X):
    """
    X must be ndarray (numpy) with dtype object

    NOTE: To obtain Uniform distribution across all choices (Categories + Typos) do: 
          -> cats_probs_list=[]
          -> typo_prob=1/(N_categories + 1)
    """

    Ns = numpy.shape(X)[0]
    ps = numpy.shape(X)[1]
    Y = numpy.copy(X)

    if self.one_cell_flag:
      # noising one cell only

      for i in range(0, Ns):

        a = numpy.random.choice(ps)

        tmp_cat_name_list = self.cats_name_lists[a] + [NoiseModel.generate_typo(str(Y[i,a]))]
        tmp_cat_prob_list = self.cats_probs_list[a]

        idx_rmv = -1
        for idx, elem in enumerate(tmp_cat_name_list):
          if elem == Y[i,a]:
            idx_rmv = idx
            break

        if idx_rmv >= 0 and len(tmp_cat_name_list) > 2:
          tmp_cat_name_list.pop(idx_rmv)
          tmp_cat_prob_list = numpy.delete(self.cats_probs_list[a], idx_rmv)
          tmp_cat_prob_list = tmp_cat_prob_list / tmp_cat_prob_list.sum()

        Y[i,a] = numpy.random.choice(tmp_cat_name_list, 1, False, tmp_cat_prob_list)[0]

    else:

      # generate idxs to be used in noising (from the selected dataset by probability)
      idxs = numpy.where(numpy.random.uniform(0.0, 1.0, X.shape) <= self.p_cell)

      # set dirty values into Y
      standard_mapping = dict(zip(range(X.shape[1]), range(X.shape[1])))
      self.corrupt_elem(Y, zip(idxs[0],idxs[1]), standard_mapping)

    return Y


  def corrupt_elem(self, Y, idxs, idx_cat_map):

    for idx_0, idx_1 in idxs:

      idx_cat = idx_cat_map[idx_1]

      tmp_cat_name_list = self.cats_name_lists[idx_cat] + [NoiseModel.generate_typo(str(Y[idx_0,idx_1]))]
      tmp_cat_prob_list = self.cats_probs_list[idx_cat]

      idx_rmv = -1
      for idx, elem in enumerate(tmp_cat_name_list):
        if elem == Y[idx_0,idx_1]:
          idx_rmv = idx
          break

      if idx_rmv >= 0 and len(tmp_cat_name_list) > 2:
        tmp_cat_name_list.pop(idx_rmv)
        tmp_cat_prob_list = numpy.delete(self.cats_probs_list[idx_cat], idx_rmv)
        tmp_cat_prob_list = tmp_cat_prob_list / tmp_cat_prob_list.sum()

      Y[idx_0,idx_1] = numpy.random.choice(tmp_cat_name_list, 1, False, tmp_cat_prob_list)[0]


"""
Simulates Mixed (Numerical and Categorical) Noise Injection
"""
class MixedNoiseModel(NoiseModel.NoiseModel):

  def __init__(self,
               shape,
               cat_array_bool,
               idx_map_cat,
               idx_map_num,
               model_list_categorical,
               model_list_numerical,
               probability=0,
               feature_importance=[],
               one_cell_flag=False,
               p_row=0.10):

    super(MixedNoiseModel, self).__init__(shape, 
                                          probability, 
                                          feature_importance,
                                          one_cell_flag)

    # noise models to be used provided as lists
    self.model_list_num = model_list_numerical
    self.model_list_cat = model_list_categorical

    # dictionary: col idx -> helper structure idx
    self.idx_map_cat = idx_map_cat
    self.idx_map_num = idx_map_num

    # one hot array (vector) to signal categorical variables
    self.cat_array_bool = cat_array_bool
    self.p_row = p_row # is like p_cell in other noise models

  def corrupt(self, X):
    """
    X must be ndarray (numpy)
    """

    Y = numpy.copy(X)

    # generate idxs to be used in noising (from the selected dataset)
    if self.one_cell_flag:
      # only choose one cell per row
      idx_mat = numpy.zeros(X.shape, dtype=bool)
      idx_mat[numpy.arange(X.shape[0]), numpy.random.choice(X.shape[1], size=X.shape[0])] = True

    else:
      # can choose several cells in the row
      idx_mat = numpy.random.uniform(0.0, 1.0, X.shape) <= self.p_row 

    # get categorical indexes for dirty cells
    idxs_cat = numpy.where(idx_mat & self.cat_array_bool)

    # get numerical indexes for dirty cells
    idxs_num = numpy.where(idx_mat & numpy.logical_not(self.cat_array_bool))

    # set dirty values into Y
    # categorical
    if len(self.model_list_cat) == 1:
      self.model_list_cat[0].corrupt_elem(Y, zip(idxs_cat[0], idxs_cat[1]), self.idx_map_cat)

    else:
      for idx_col, model_cat in zip(self.idx_map_cat.keys(), self.model_list_cat):

        # filter to get only current column
        cur_idx_list = [elem for elem in zip(idxs_cat[0], idxs_cat[1]) if elem[1] == idx_col]

        model_cat.corrupt_elem(Y, cur_idx_list, self.idx_map_cat)

    # numerical
    if len(self.model_list_num) == 1:
      self.model_list_num[0].corrupt_elem(Y, zip(idxs_num[0], idxs_num[1]), self.idx_map_num)

    else:
      for idx_col, model_num in zip(self.idx_map_num.keys(), self.model_list_num):

        # filter to get only current column
        cur_idx_list = [elem for elem in zip(idxs_num[0], idxs_num[1]) if elem[1] == idx_col]

        model_num.corrupt_elem(Y, cur_idx_list, self.idx_map_num)

    return Y


"""
Simulates Random Missing Data With a Placeholder Value
Picks an attr at random and sets the value to be missing
"""
class MissingNoiseModel(NoiseModel.NoiseModel):

  """
  ph is the Placeholder value that missing attrs are set to.
  """
  def __init__(self, 
               shape, 
               probability=0,
               feature_importance=[],
               ph=-1):

    super(MissingNoiseModel, self).__init__(shape, 
                                   probability, 
                                   feature_importance,
                                   True)
    self.ph = ph

  def corrupt(self, X):
    Ns = numpy.shape(X)[0]
    ps = numpy.shape(X)[1]
    Y = numpy.copy(X)
    for i in range(0, Ns):
      a = numpy.random.choice(ps,1)
      Y[i,a] = self.ph
    return Y


"""
Adding Salt and Pepper noise to the image dataset:
  - Assumes image dataset is standardized between 0-1 magnitude grayscale
  - Picks pixels (cells) at random, replaces pixel value by 0 or 1
  - 0 and 1 can be replaced by min_val and max_val
"""
class ImageSaltnPepper(NoiseModel.NoiseModel):

  def __init__(self,
               shape,
               probability=0,
               feature_importance=[], 
               one_cell_flag=False, 
               min_val=0.,
               max_val=1.,
               p_min=0.5,
               p_pixel=-1,
               conv_to_int=False):

    # NOTE: one_cell_flag <=> one pixel flag
    #       p_pixel <=> p_cell

    super(ImageSaltnPepper, self).__init__(shape,
                                           probability,
                                           feature_importance,
                                           one_cell_flag)

    self.p_pixel = p_pixel
    self.min_val = min_val
    self.max_val = max_val
    self.p_min = p_min

    self.x_dim = shape[1] # image dimensions
    self.y_dim = shape[2]

    self.conv_to_int = conv_to_int

  def corrupt(self, X):

    Ns = numpy.shape(X)[0]

    Y = numpy.copy(X)
    Y = Y.reshape((Ns,-1))

    ps = numpy.shape(Y)[1]

    if not self.one_cell_flag: # noise several pixels (standard case)

      Z = numpy.random.choice([self.min_val, self.max_val], 
                              size=(Ns,ps), replace=True, 
                              p=[self.p_min,1.-self.p_min])

      if self.conv_to_int:
        Z = Z.astype(int)

      if self.p_pixel <= 0:
        mask_mtx = numpy.ones((Ns, ps))

      else:
        mask_mtx = numpy.random.uniform(0.0, 1.0, (Ns, ps)) <= self.p_pixel

      Y[mask_mtx] = Z[mask_mtx]

      return vbound(Y, numpy.finfo(float).min, numpy.finfo(float).max).reshape(X.shape)

    else: # noise one pixel only

      for i in range(0, Ns):
        a = numpy.random.choice(ps)

        noiz = numpy.random.choice([self.min_val, self.max_val], 
                                   size=1, replace=True, 
                                   p=[self.p_min,1.-self.p_min])

        if self.conv_to_int:
          noiz = noiz.astype(int)

        Y[i,a] = bound_number(noiz, numpy.finfo(float).min, numpy.finfo(float).max)

      return Y.reshape(X.shape)

class ImageAdditiveGaussianNoise(NoiseModel.NoiseModel):

  def __init__(self,
               shape,
               probability=0,
               feature_importance=[],
               one_cell_flag=False,
               min_val=0.,
               max_val=1.,
               mu=0.,
               sigma=1.,
               scale=numpy.array([]),
               p_pixel=-1):

    # NOTE: one_cell_flag <=> one pixel flag
    #       p_pixel <=> p_cell

    super(ImageAdditiveGaussianNoise, self).__init__(shape,
                                                     probability,
                                                     feature_importance,
                                                     one_cell_flag)

    # NOTE: scale and mu are shared amongst pixels

    self.p_pixel = p_pixel
    self.min_val = min_val
    self.max_val = max_val
    self.scale = scale
    self.sigma = sigma
    self.mu = mu

    self.x_dim = shape[1] # image dimensions
    self.y_dim = shape[2]

  def corrupt(self, X):

    Ns = numpy.shape(X)[0]
    ps = numpy.shape(X)[1]

    if not self.one_cell_flag: # noise several pixels (standard case)

      if not self.scale.size:
        scale = 1.0
      else:
        scale = self.scale

      Z = numpy.random.randn(Ns,ps)*self.sigma*scale + self.mu

      if self.p_pixel <= 0:
        mask_mtx = numpy.ones((Ns, ps))
      else:
        mask_mtx = numpy.random.uniform(0.0, 1.0, (Ns, ps)) <= self.p_pixel
      Z = Z * mask_mtx

      Y = numpy.clip((X + Z), self.min_val, self.max_val)

      return vbound(Y, numpy.finfo(float).min, numpy.finfo(float).max)

    else: # noise one pixel only

      Y = numpy.copy(X)

      for i in range(0, Ns):
        a = numpy.random.choice(ps)

        if not self.scale.size:
          scale = 1.0
        else:
          scale = self.scale

        noiz = numpy.random.randn()*self.sigma*scale + self.mu
        Y[i,a] = bound_number(Y[i,a] + noiz, numpy.finfo(float).min, numpy.finfo(float).max)
        Y[i,a] = numpy.clip(Y[i,a], self.min_val, self.max_val)

      return Y
