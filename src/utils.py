from data_reader import ABIDEDataReader
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import slayer

NUM_ROIS = 200


class DataContainer:

    def __init__(self, data):
        self.X_train = data[0]
        self.X_test  = data[1]
        self.y_train = data[2]
        self.y_test  = data[3]


def read_and_build_features(fresh=False, compute_features=True, sample=None):
    """
    Read the data from disk and compute features (persistence diagram, landscape
    and image. Stores the computed list to disk in `data_processed` directory

    :param fresh: If False, look for precomputed data in `data_processed` folder,
        otherwise compute new features
    :param compute_features: If True, compute persistence features for each individual
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
            subjects = reader.read(raw_data_location, num_rois=NUM_ROIS,
                                   compute_features=compute_features, sample=sample)

            with open(dump_location, 'wb') as f:
                pickle.dump(subjects, f)

        return subjects

    else:
        reader = ABIDEDataReader()
        subjects = reader.read(raw_data_location, num_rois=NUM_ROIS,
                               compute_features=compute_features, sample=sample)

        with open(dump_location, 'wb') as f:
            pickle.dump(subjects, f)

        return subjects


def split_train_test(complete_data, test_size):
    X = complete_data
    Y = np.array([d.label for d in complete_data])
    return DataContainer(train_test_split(X, Y, test_size=test_size))


def get_corr_features(X):
    return np.vstack([x.corr_vector for x in X]).astype(np.float32)


def get_pers_img_features(X):
    return np.vstack([x.persistence_image for x in X]).astype(np.float32)


def get_pers_landscape_features(X):
    return np.vstack([x.persistence_landscape for x in X]).astype(np.float32)


def get_topology_features(X, dims=(0, 1)):
    topology_features = []

    for dim in dims:
        topology_features.append([torch.Tensor(x.persistence_diagram[dim]) for x in X])

    return topology_features


def get_hybrid_features(X):

    topo_features = get_topology_features(X)

    data_dict = {'pers_dim0': slayer.prepare_batch(topo_features[0])[0],
                 'pers_dim1': slayer.prepare_batch(topo_features[1])[0],
                 'corr_features': torch.Tensor(get_corr_features(X))}

    return data_dict


def get_pers_img_corr_features(X):

    data_dict = {'vec_feature_1': torch.Tensor(get_pers_img_features(X)),
                 'vec_feature_2': torch.Tensor(get_corr_features(X))}

    return data_dict


def score(model, X, ground_truth, metric=accuracy_score):
    return metric(model.predict(X), ground_truth)


def get_pi_features(X):
    return np.vstack([x.persistence_image for x in X]).astype(np.float32)


def get_pers_landscape_corr_features(X):
    data_dict = {'vec_feature_1': torch.Tensor(get_pers_landscape_features(X)),
                 'vec_feature_2': torch.Tensor(get_corr_features(X))}

    return data_dict


def get_pers_diag_features(X):
    topo_features = get_topology_features(X)

    data_dict = {'pers_dim0': slayer.prepare_batch(topo_features[0])[0],
                 'pers_dim1': slayer.prepare_batch(topo_features[1])[0]}

    return data_dict


def trainer(model, data_container: DataContainer, features='correlation'):
    if features == 'correlation':
        X_train = get_corr_features(data_container.X_train)
        X_test = get_corr_features(data_container.X_test)
    elif features == 'pers_img':
        X_train = get_pi_features(data_container.X_train)
        X_test  = get_pi_features(data_container.X_test)
    elif features == 'pers_img_corr':
        X_train = get_pers_img_corr_features(data_container.X_train)
        X_test = get_pers_img_corr_features(data_container.X_test)
    elif features == 'pers_landscape':
        X_train = get_pers_landscape_features(data_container.X_train)
        X_test = get_pers_landscape_features(data_container.X_test)
    elif features == 'pers_landscape_corr':
        X_train = get_pers_landscape_corr_features(data_container.X_train)
        X_test = get_pers_landscape_corr_features(data_container.X_test)
    elif features == 'kernel':
        X_train = data_container.X_train
        X_test = data_container.X_test
    elif features == 'pers_diag':
        X_train = get_pers_diag_features(data_container.X_train)
        X_test = get_pers_diag_features(data_container.X_test)

    model.fit(X_train, data_container.y_train)
    print(score(model, X_test, data_container.y_test))
