from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
import sklearn_tda


class NNCorr(nn.Module):
    pass


class NNCorrPlusTDA(nn.Module):
    pass


def get_kernel(kernel='scale_space', weights=(0.5, 0.5)):
    """
    Return a kernel function for use in kernel methods
    :param kernel:
    :return:
    """


    kernels = {
        'scale_space': sklearn_tda.PersistenceScaleSpaceKernel(),
        'weighted_gaussian': sklearn_tda.PersistenceWeightedGaussianKernel(),
        'sliced_wasserstein': sklearn_tda.SlicedWassersteinKernel(),
        'fisher': sklearn_tda.PersistenceFisherKernel(),
    }

    if kernel not in kernels.keys():
        raise KeyError("Specified kernel not found. Make sure it is one "
                       "of ['scale_space', 'weighted_gaussian', 'slice_wasserstein', 'fisher]")

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
