from skorch import NeuralNetClassifier

import utils
import models

from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt


# Script constants
TEST_FRACTION = 0.20
EPOCHS        = 100

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

corr_models = {'svm_corr'          : SVC(kernel='linear'),
               'random_forest_corr': RandomForestClassifier(n_estimators=500, max_depth=5),
               'neural_net_corr'   : NeuralNetClassifier(models.NNCorr([corr_feature_size, 10, 2]),
                                                         max_epochs=EPOCHS, verbose=False,
                                                         warm_start=True)}


pi_models = {'svm_pi'           : SVC(kernel='linear'),
             'random_forest_pi' : RandomForestClassifier(n_estimators=500, max_depth=5),
             'neural_net_pi'    : NeuralNetClassifier(models.NNCorr([pi_feature_size, 10, 2]),
                                                      max_epochs=EPOCHS, verbose=False,
                                                      warm_start=True)}


pl_models = {'svm_pl'           : SVC(kernel='linear'),
             'random_forest_pl' : RandomForestClassifier(n_estimators=500, max_depth=5),
             'neural_net_pl'    : NeuralNetClassifier(models.NNCorr([pl_feature_size, 10, 2]),
                                                      max_epochs=EPOCHS, verbose=False,
                                                      warm_start=True)}

kernel_models = {'svm_scale_space'       : models.PersistenceKernelSVM(kernel_type='scale_space'),
                 'svm_weighted_gaussian' : models.PersistenceKernelSVM(kernel_type='weighted_gaussian'),
                 'svm_sliced_wasserstein': models.PersistenceKernelSVM(kernel_type='sliced_wasserstein'),
                 'svm_fisher'            : models.PersistenceKernelSVM(kernel_type='fisher')}

pd_models = {'neural_net_pd' : NeuralNetClassifier(models.NNPersDiag([[pers_input_size, 25], [pers_input_size, 25]], [50, 2]),
                                                   max_epochs=EPOCHS, verbose=False,
                                                   warm_start=True)}

# hybrid_models = {'pd_corr' : NeuralNetClassifier(models.NNHybridPers([[pers_input_size, 25], [pers_input_size, 25]], [corr_feature_size, 500, 25], [75, 2]),
#                                                                      max_epochs=EPOCHS,
#                                                                      warm_start=True),
#                  'pi_corr' : NeuralNetClassifier(models.NNHybridVec([[pi_feature_size, 10], [corr_feature_size, 10]], [20, 10, 2]),
#                                                  max_epochs=EPOCHS,
#                                                  warm_start=True),
#                  'pl_corr' : NeuralNetClassifier(models.NNHybridVec([[pl_feature_size, 10], [corr_feature_size, 10]], [20, 10, 2]),
#                                                  max_epochs=EPOCHS,
#                                                  warm_start=True)}

feature_wise_models = {'correlation'    : corr_models,
                       'pers_img'       : pi_models,
                       'pers_landscape' : pl_models,
                       'pers_diag'      : pd_models}
                       # 'kernel'         : kernel_models}

for feature_type, model_dict in feature_wise_models.items():
    for model_name, model in model_dict.items():
        print(model_name)
        utils.trainer(model, dataset, features=feature_type)

exit()
# Correlation based models

# Correlation SVM
svm_corr = SVC(kernel='linear')
# utils.trainer(svm_corr, dataset, features='correlation')

# Correlation Random-Forest
random_forest_corr = RandomForestClassifier(n_estimators=100, max_depth=3)
# utils.trainer(random_forest_corr, dataset, features='correlation')

# Correlation Neural Network
corr_feature_size = dataset.X_train[0].corr_vector.shape[0]

neural_net_corr = models.NNCorr([corr_feature_size, 10, 2])
net_corr = NeuralNetClassifier(neural_net_corr, max_epochs=20, warm_start=True)
# utils.trainer(net, dataset, features='correlation')

# Save models and predictions

# -----------------------------------

# Vector based topological features

# Get Persistence image features for train and test sets

# PI SVM
svm_pi = LinearSVC()
# utils.trainer(svm_pi, dataset, features='pers_img')

# PI Random-Forest
random_forest_pi = RandomForestClassifier(n_estimators=500, max_depth=5)
# utils.trainer(random_forest_pi, dataset, features='pers_img')

# PI Neural Network
pi_feature_size = dataset.X_train[0].persistence_image.shape[0]

neural_net_pi = models.NNCorr([pi_feature_size, 10, 2])
net_pi = NeuralNetClassifier(neural_net_pi, max_epochs=200, warm_start=True)
# utils.trainer(net_pi, dataset, features='pers_img')

# PI + Corr Neural Network
neural_net_pi_corr = models.NNHybridVec([[pi_feature_size, 10], [corr_feature_size, 10]], [20, 10, 2])
net_pi_corr = NeuralNetClassifier(neural_net_pi_corr, max_epochs=200, warm_start=True)
# utils.trainer(net_pi_corr, dataset, features='pers_img_corr')

# Get Persistence landscape features for train and test sets

# PL SVM
svm_pl = LinearSVC()
# utils.trainer(svm_pl, dataset, features='pers_landscape')

# PL Random-Forest
random_forest_pl = RandomForestClassifier(n_estimators=500, max_depth=5)
# utils.trainer(random_forest_pl, dataset, features='pers_landscape')

# PL Neural Network
pl_feature_size = dataset.X_train[0].persistence_landscape.shape[0]

neural_net_pl = models.NNCorr([pl_feature_size, 10, 2])
net_pl = NeuralNetClassifier(neural_net_pl, max_epochs=200, warm_start=True)
# utils.trainer(net_pl, dataset, features='pers_landscape')

# PL + Corr Neural Network
neural_net_pl_corr = models.NNHybridVec([[pl_feature_size, 50, 10], [corr_feature_size, 500, 10]], [20, 10, 2])
net_pl_corr = NeuralNetClassifier(neural_net_pl_corr, max_epochs=100, warm_start=True, optimizer__weight_decay=0.01)
# utils.trainer(net_pl_corr, dataset, features='pers_landscape_corr')

# Save models and predictions

# ----------------------------------

# Topological features based methods

# Kernels - SVM
# svm_scale_space_kernel = models.PersistenceKernelSVM(kernel_type='scale_space')
# utils.trainer(svm_scale_space_kernel, dataset, features='kernel')
#
# svm_weighted_gaussian_kernel = models.PersistenceKernelSVM(kernel_type='weighted_gaussian')
# utils.trainer(svm_weighted_gaussian_kernel, dataset, features='kernel')
#
# svm_sliced_wasserstein_kernel = models.PersistenceKernelSVM(kernel_type='sliced_wasserstein')
# utils.trainer(svm_sliced_wasserstein_kernel, dataset, features='kernel')

svm_fisher_kernel = models.PersistenceKernelSVM(kernel_type='fisher')
utils.trainer(svm_fisher_kernel, dataset, features='kernel')

# Topological Neural Network model


# Topo + Corr Neural Network

# Save models and predictions

# ----------------------------------
