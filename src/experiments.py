from skorch import NeuralNetClassifier

import utils
import models

from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt


# Script constants
TEST_FRACTION = 0.20
EPOCHS        = 20

# Read data
data = utils.read_and_build_features()
# data = utils.read_and_build_features()[:50]  # smaller data for testing purposes

# Split into train-test
dataset = utils.split_train_test(data, TEST_FRACTION)

# -----------------------------------
corr_feature_size = dataset.X_train[0].corr_vector.shape[0]
pi_feature_size   = dataset.X_train[0].persistence_image.shape[0]
pl_feature_size   = dataset.X_train[0].persistence_landscape.shape[0]
pers_input_size   = 50

corr_models = {'svm_corr' : SVC(kernel='linear'),
               'rf_corr'  : RandomForestClassifier(n_estimators=500, max_depth=5),
               'nn_corr'  : NeuralNetClassifier(models.NNVec([corr_feature_size, 10, 2]),
                                                max_epochs=EPOCHS, verbose=False, warm_start=True)}


pi_models = {'svm_pi' : SVC(kernel='linear'),
             'rf_pi'  : RandomForestClassifier(n_estimators=500, max_depth=5),
             'nn_pi'  : NeuralNetClassifier(models.NNVec([pi_feature_size, 10, 2]),
                                            max_epochs=EPOCHS, verbose=False, warm_start=True)}


pl_models = {'svm_pl'           : SVC(kernel='linear'),
             'random_forest_pl' : RandomForestClassifier(n_estimators=500, max_depth=5),
             'neural_net_pl'    : NeuralNetClassifier(models.NNVec([pl_feature_size, 10, 2]),
                                                      max_epochs=EPOCHS, verbose=False, warm_start=True)}

kernel_models = {'svm_scale_space'       : models.PersistenceKernelSVM(kernel_type='scale_space'),
                 'svm_weighted_gaussian' : models.PersistenceKernelSVM(kernel_type='weighted_gaussian'),
                 'svm_sliced_wasserstein': models.PersistenceKernelSVM(kernel_type='sliced_wasserstein'),
                 'svm_fisher'            : models.PersistenceKernelSVM(kernel_type='fisher')}

pd_models = {'nn_pd' : NeuralNetClassifier(models.NNPersDiag([[pers_input_size, 25], [pers_input_size, 25]], [50, 2]),
                                                   max_epochs=EPOCHS, verbose=False, warm_start=True)}

hybrid_models = {'pd_corr' : NeuralNetClassifier(models.NNHybridPers([[pers_input_size, 25], [pers_input_size, 25]], [corr_feature_size, 500, 25], [75, 2]),
                                                        max_epochs=EPOCHS, verbose=False, warm_start=True),
                 'pi_corr' : NeuralNetClassifier(models.NNHybridVec([[pi_feature_size, 10], [corr_feature_size, 10]], [20, 10, 2]),
                                                        max_epochs=EPOCHS, verbose=False, warm_start=True),
                 'pl_corr' : NeuralNetClassifier(models.NNHybridVec([[pl_feature_size, 10], [corr_feature_size, 10]], [20, 10, 2]),
                                                        max_epochs=EPOCHS, verbose=False, warm_start=True)}

feature_wise_models = {'correlation' : corr_models,
                       'pi'          : pi_models,
                       'pl'          : pl_models,
                       'pd'          : pd_models,
                       'hybrid'      : hybrid_models}

for feature_type, model_dict in feature_wise_models.items():
    for model_name, model in model_dict.items():
        print(model_name)
        if feature_type == 'hybrid':
            utils.trainer(model, dataset, model_name, features=model_name)
        else:
            utils.trainer(model, dataset, model_name, features=feature_type)
