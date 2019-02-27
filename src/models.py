import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
import sklearn_tda
import warnings
from slayer import SLayer, UpperDiagonalThresholdedLogTransform
import numpy as np


def create_linear(layer_shapes, dropout_prob):
    linear = nn.Sequential()

    for layer_num, (num_in, num_out) in enumerate(zip(layer_shapes, layer_shapes[1:])):
        linear.add_module('linear_{}'.format(layer_num), nn.Linear(num_in, num_out))
        linear.add_module('batchnorm_{}'.format(layer_num), nn.BatchNorm1d(num_out))
        if layer_num != len(layer_shapes) - 2:
            linear.add_module('dropout_{}'.format(layer_num), nn.Dropout(dropout_prob))
            linear.add_module('relu_{}'.format(layer_num), nn.ReLU())

    return linear


def create_slayer_linear(layer_shapes, dropout_prob):
    linear = nn.Sequential()

    for layer_num, (num_in, num_out) in enumerate(zip(layer_shapes, layer_shapes[1:])):
        if layer_num == 0:
            linear.add_module('slayer_{}'.format(layer_num), SLayer(num_in, 2))

        linear.add_module('linear_{}'.format(layer_num), nn.Linear(num_in, num_out))
        linear.add_module('batchnorm_{}'.format(layer_num), nn.BatchNorm1d(num_out))

        if layer_num != len(layer_shapes) - 2:
            linear.add_module('dropout_{}'.format(layer_num), nn.Dropout(dropout_prob))
            linear.add_module('relu_{}'.format(layer_num), nn.ReLU())

    return linear


class NNCorr(nn.Module):
    def __init__(self, layer_shapes, dropout_prob=0.1):
        super(NNCorr, self).__init__()
        self.linear = create_linear(layer_shapes, dropout_prob)

    def forward(self, batch):
        # x = torch.stack(batch)
        x = nn.functional.relu(self.linear(batch))

        return nn.functional.softmax(x, dim=1)


class NNPersDiag(nn.Module):

    def __init__(self, pers_layers_shapes, merge_layer_shape, dropout_prob=0.1):
        super(NNPersDiag, self).__init__()

        self.branches = [create_slayer_linear(branch_shape, dropout_prob) for branch_shape in pers_layers_shapes]

        for i, branch in enumerate(self.branches):
            self.add_module('branch_{}'.format(i), branch)

        if sum(map(lambda x: x[-1], pers_layers_shapes)) != merge_layer_shape[0]:
            warnings.warn('Layer shape mismatch between first layer of merge and last layers of parallel modules',
                          UserWarning)

        self.merge_layer = create_linear(merge_layer_shape, dropout_prob)

    def forward(self, pers_dim0, pers_dim1):
        inputs = [pers_dim0, pers_dim1]
        x = torch.cat([branch(branch_input) for branch, branch_input in zip(self.branches, inputs)], 1)
        x = self.merge_layer(x)

        return nn.functional.softmax(x, dim=1)


class NNHybridVec(nn.Module):
    def __init__(self, branchwise_shapes, merge_layer_shape, dropout_prob=0.1):
        super(NNHybridVec, self).__init__()

        self.branches = [create_linear(branch_shape, dropout_prob) for branch_shape in branchwise_shapes]

        for i, branch in enumerate(self.branches):
            self.add_module('branch_{}'.format(i), branch)

        if sum(map(lambda x: x[-1], branchwise_shapes)) != merge_layer_shape[0]:
            warnings.warn('Layer shape mismatch between first layer of merge and last layers of parallel modules',
                          UserWarning)

        self.merge_layer = create_linear(merge_layer_shape, dropout_prob)

    def forward(self, vec_feature_1, vec_feature_2):
        inputs = [vec_feature_1, vec_feature_2]
        x = torch.cat([branch(branch_input) for branch, branch_input in zip(self.branches, inputs)], 1)
        x = self.merge_layer(x)

        return nn.functional.softmax(x, dim=1)


class NNHybridPers(nn.Module):
    def __init__(self, pers_layers_shapes, vec_layer_shapes, merge_layer_shape, dropout_prob=0.1):
        super(NNHybridPers, self).__init__()

        self.branches = [create_slayer_linear(branch_shape, dropout_prob) for branch_shape in pers_layers_shapes]
        self.branches += [create_linear(branch_shape, dropout_prob) for branch_shape in vec_layer_shapes]

        for i, branch in enumerate(self.branches):
            self.add_module('branch_{}'.format(i), branch)

        if sum(map(lambda x: x[-1], pers_layers_shapes + vec_layer_shapes)) != merge_layer_shape[0]:
            warnings.warn('Layer shape mismatch between first layer of merge and last layers of parallel modules',
                          UserWarning)

        self.merge_layer = create_linear(merge_layer_shape, dropout_prob)

    def forward(self, pers_dim0, pers_dim1, corr_features):
        inputs = [pers_dim0, pers_dim1, corr_features]
        x = torch.cat([branch(branch_input) for branch, branch_input in zip(self.branches, inputs)], 1)
        x = self.merge_layer(x)

        return nn.functional.softmax(x, dim=1)


def get_kernel(kernel='scale_space', weights=(0.5, 0.5)):
    """
    Return a kernel function for use in kernel methods
    :param kernel: type of persistence kernel to use, should be one of 'scale space',
        'weighted gaussian', 'sliced_wasserstein' or 'fisher'. The same kernel is used
        across all homology dimensions
    :param weights: scalar factor to weigh each dimension's gram matrix
    :return: sum of weighted gram matrices
    """

    # kernel_approx = RBFSampler(gamma=0.5, n_components=100000).fit(np.ones([1,2]))
    kernel_approx = None

    kernels = {
        'scale_space': sklearn_tda.PersistenceScaleSpaceKernel(kernel_approx=kernel_approx),
        'weighted_gaussian': sklearn_tda.PersistenceWeightedGaussianKernel(kernel_approx=kernel_approx),
        'sliced_wasserstein': sklearn_tda.SlicedWassersteinKernel(),
        'fisher': sklearn_tda.PersistenceFisherKernel(kernel_approx=kernel_approx),
    }

    if kernel not in kernels.keys():
        raise KeyError("Specified kernel not found. Make sure it is one "
                       "of ['scale_space', 'weighted_gaussian', 'sliced_wasserstein', 'fisher']")

    k = kernels[kernel]

    def kernel_wrapper(data1, data2):

        kernel_matrices = []

        for dim, w in enumerate(weights):
            X = [x.persistence_diagram[dim] for x in data1]
            Y = [y.persistence_diagram[dim] for y in data2]
            k_matrix = w * k.fit(X).transform(Y)
            kernel_matrices.append(k_matrix)

        return sum(kernel_matrices)

    return kernel_wrapper


class PersistenceKernelSVM(BaseEstimator, ClassifierMixin):
    """
    A sklearn compatible SVM classifier that computes persistent homology based kernels

    :param kernel_type: type of persistence kernel to use, should be one of 'scale space',
        'weighted gaussian', 'sliced_wasserstein' or 'fisher'. The same kernel is used
        across all homology dimensions
    :param C: traditional `C` parameter for SVMs
    :param homology_dims: the dimensions for which kernel is evaluated
    :param hdim_weights: the weights for kernels of each dimensions. Should be of same
        length as `homology_dims`
    """

    def __init__(self, kernel_type='scale_space', C=1.0, homology_dims=(0, 1), hdim_weights=(0.5, 0.5)):

        self.kernel_type = kernel_type
        self.C = C
        self.homology_dims = homology_dims
        self.hdim_weights = hdim_weights

    def fit(self, X, y):
        self.X_ = X

        # Check that weights sum to one
        if sum(self.hdim_weights) != 1:
            raise ValueError('hdim_weights = {} do not sum to 1.'.format(self.hdim_weights))

        # Get the kernel function as a sum of kernel for each homology dimension
        self.kernel_ = get_kernel(self.kernel_type, self.hdim_weights)

        self.svm = SVC(C=self.C, kernel='precomputed')
        self.svm.fit(self.kernel_(X, X), y)

        return self

    def predict(self, X):
        return self.svm.predict(self.kernel_(self.X_, X))
