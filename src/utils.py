from data_reader import ABIDEDataReader
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn_tda
import numpy as np
import torch
import models
import slayer
from skorch import NeuralNetClassifier

NUM_ROIS = 200


def read_and_build_features(fresh=False, compute_features=True):
    """
    Read the data from disk and compute features (persistence diagram, landscape
    and image. Stores the computed list to disk in `data_processed` directory

    :param fresh: If False, look for precomputed data in `data_processed` folder,
        otherwise compute new features
    :return: list of Subject class objects that contain individual data and
        computed features
    """
    raw_data_location = '../data/ABIDE/rois_cc{}'.format(NUM_ROIS)
    dump_location = '../data_processed/subjects_abide{}.pkl'.format(NUM_ROIS)

    if not fresh:
        # Load the precomputed results if it exists already
        try:
            with open(dump_location, 'rb') as f:
                subjects = pickle.load(f)

        except FileNotFoundError:
            reader = ABIDEDataReader()
            subjects = reader.read(raw_data_location, num_rois=NUM_ROIS, compute_features=compute_features)

            with open(dump_location, 'wb') as f:
                pickle.dump(subjects, f)

        return subjects

    else:
        reader = ABIDEDataReader()
        subjects = reader.read(raw_data_location, num_rois=NUM_ROIS, compute_features=compute_features)

        with open(dump_location, 'wb') as f:
            pickle.dump(subjects, f)

        return subjects


def split_train_test(complete_data, test_size):
    X = complete_data
    Y = np.array([d.label for d in complete_data])
    return train_test_split(X, Y, test_size=test_size)


def get_corr_features(X):
    return np.vstack([x.corr_vector for x in X]).astype(np.float32)


def get_topology_features(X, dims=(0, 1)):
    topology_features = []

    for dim in dims:
        topology_features.append([torch.Tensor(x.persistence_diagram[dim]) for x in X])

    return topology_features


def get_hybrid_features(X):
    data_dict = {}

    topo_features = get_topology_features(X)
    data_dict['pers_dim0'] = slayer.prepare_batch(topo_features[0])[0]
    data_dict['pers_dim1'] = slayer.prepare_batch(topo_features[1])[0]

    data_dict['corr_features'] = torch.Tensor(get_corr_features(X))
    return data_dict


def score(model, X, ground_truth, metric=accuracy_score):
    return metric(model.predict(X), ground_truth)


# # Read data and compute persistence diagram
# data = read_and_build_features()
#
# # Split into training and testing sets
# X_train, X_test, y_train, y_test = split_train_test(data, 0.98)
#
# # scale_space_svm = models.PersistenceKernelSVM(kernel_type='scale_space')
# # scale_space_svm.fit(X_train, y_train)
# # print(scale_space_svm.score(X_test, y_test))
#
# # sliced_wasserstein_svm = models.PersistenceKernelSVM(kernel_type='sliced_wasserstein')
# # sliced_wasserstein_svm.fit(X_train, y_train)
# # print(sliced_wasserstein_svm.score(X_test, y_test))
# #
# X_train_hybrid = get_hybrid_features(X_train)
# X_test_hybrid = get_hybrid_features(X_test)
#
# nn_hybrid = models.NNHybridPers([[50, 25], [50, 25]], [[X_train_hybrid['corr_features'].shape[1], 5000, 500, 25]], [75, 25, 10, 2])
# # net = NeuralNetClassifier(nn_hybrid, max_epochs=5, criterion=torch.nn.CrossEntropyLoss, train_split=None)
# #
# # net.fit(X_train_hybrid, y_train)
#
# nn_hybrid(*X_train_hybrid)