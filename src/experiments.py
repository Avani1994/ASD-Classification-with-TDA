from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from skorch import NeuralNetClassifier

import utils
from models import *

TEST_FRACTION = 0.20
EPOCHS        = 20

# Read data
# data = utils.read_and_build_features()
data = utils.read_and_build_features()[:50]  # smaller data for testing purposes

# Split into train-test
dataset = utils.split_train_test(data, TEST_FRACTION)

corr_feature_size = dataset.X_train[0].corr_vector.shape[0]
pi_feature_size   = dataset.X_train[0].persistence_image.shape[0]
pl_feature_size   = dataset.X_train[0].persistence_landscape.shape[0]
pers_input_size   = 50

modelManager = ModelManager('../data_processed', dataset, overwrite=True)

featureExtractors = {'corr': utils.get_corr_features,
                     'pi_corr': utils.get_pers_img_corr_features,
                     'pl_corr': utils.get_pers_landscape_corr_features,
                     'pd_corr': utils.get_pers_diag_corr_features,
                     'pi': utils.get_pers_img_features,
                     'pl': utils.get_pers_landscape_features,
                     'pd': utils.get_pers_diag_features,
                     'pd_kern': utils.get_pers_diag_kern_features
                    }

svm_corr = SVC(kernel='linear')
rf_corr = RandomForestClassifier(n_estimators=500, max_depth=5)
nn_corr = NeuralNetClassifier(NNVec([corr_feature_size, 100, 2], dropout_prob=0.5),
                              max_epochs=EPOCHS, verbose=False, warm_start=True)

modelManager.add_model(svm_corr, 'svm_corr', featureExtractors['corr'])
modelManager.add_model(rf_corr, 'rf_corr', featureExtractors['corr'])
modelManager.add_model(nn_corr, 'nn_corr', featureExtractors['corr'])

# Persistence image models


svm_pi = SVC(kernel='linear')
rf_pi = RandomForestClassifier(n_estimators=500, max_depth=5)
nn_pi = NeuralNetClassifier(NNVec([pi_feature_size, 10, 2]),
                            max_epochs=EPOCHS, verbose=False, warm_start=True)

modelManager.add_model(svm_pi, 'svm_pi', featureExtractors['pi'])
modelManager.add_model(rf_pi, 'rf_pi', featureExtractors['pi'])
modelManager.add_model(nn_pi, 'nn_pi', featureExtractors['pi'])

# ### Persistence Landscape models

svm_pl = SVC(kernel='linear')
rf_pl = RandomForestClassifier(n_estimators=500, max_depth=5)
nn_pl = NeuralNetClassifier(NNVec([pl_feature_size, 10, 2], dropout_prob=0.5),
                            max_epochs=EPOCHS, verbose=False, warm_start=True)

modelManager.add_model(svm_pl, 'svm_pl', featureExtractors['pl'])
modelManager.add_model(rf_pl, 'rf_pl', featureExtractors['pl'])
modelManager.add_model(nn_pl, 'nn_pl', featureExtractors['pl'])

# Persistence diagram models
nn_pd = NeuralNetClassifier(NNPersDiag([[pers_input_size, 25], [pers_input_size, 25]], [50, 2], dropout_prob=0.5),
                            max_epochs=EPOCHS, verbose=False, warm_start=True)

modelManager.add_model(nn_pd, 'nn_pd', featureExtractors['pd'])

# Hybrid models
pd_corr = NeuralNetClassifier(
    NNHybridPers([[pers_input_size, 25], [pers_input_size, 25]], [corr_feature_size, 500, 25], [75, 2],
                 dropout_prob=0.5),
    max_epochs=EPOCHS, verbose=False, warm_start=True)
pi_corr = NeuralNetClassifier(
    NNHybridVec([[pi_feature_size, 10], [corr_feature_size, 10]], [20, 10, 2], dropout_prob=0.5),
    max_epochs=EPOCHS, verbose=False, warm_start=True)
pl_corr = NeuralNetClassifier(
    NNHybridVec([[pl_feature_size, 10], [corr_feature_size, 10]], [20, 10, 2], dropout_prob=0.5),
    max_epochs=EPOCHS, verbose=False, warm_start=True)

modelManager.add_model(pd_corr, 'pd_corr', featureExtractors['pd_corr'])
modelManager.add_model(pi_corr, 'pi_corr', featureExtractors['pi_corr'])
modelManager.add_model(pl_corr, 'pl_corr', featureExtractors['pl_corr'])

# Topological kernels
svm_scalespace = PersistenceKernelSVM(kernel_type='scale_space')
svm_weightedgaussian = PersistenceKernelSVM(kernel_type='weighted_gaussian')
svm_slicedwasserstein = PersistenceKernelSVM(kernel_type='sliced_wasserstein')
svm_fisher = PersistenceKernelSVM(kernel_type='fisher')

# modelManager.add_model(svm_scalespace, 'svm_scalespace', featureExtractors['pd_kern'])
# modelManager.add_model(svm_weightedgaussian, 'svm_weightedgaussian', featureExtractors['pd_kern'])
# modelManager.add_model(svm_slicedwasserstein, 'svm_slicedwasserstein', featureExtractors['pd_kern'])
# modelManager.add_model(svm_fisher, 'svm_fisher', featureExtractors['pd_kern'])

modelManager.train_all()
modelManager.evaluate_all(accuracy_score)
print(modelManager.tabulate())

import pickle

with open('../models/modelManager.pkl', 'wb') as f:
    pickle.dump(modelManager, f)
