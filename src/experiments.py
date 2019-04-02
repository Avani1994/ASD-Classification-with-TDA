import matplotlib.pyplot as plt
import utils
from models import *
from sklearn.metrics import accuracy_score

from skorch.dataset import Dataset
from skorch import NeuralNetClassifier, callbacks
from skorch.helper import predefined_split

from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt


TEST_FRACTION = 0.20
EPOCHS        = 200

# Read data
data = utils.read_and_build_features()
# data = utils.read_and_build_features()[:50]; EPOCHS=20  # smaller data for testing purposes

# Split into train-test
dataset = utils.split_train_test(data, TEST_FRACTION)

corr_feature_size = dataset.X_train[0].corr_vector.shape[0]
pi_feature_size   = dataset.X_train[0].persistence_image.shape[0]
pl_feature_size   = dataset.X_train[0].persistence_landscape.shape[0]
pers_input_size   = 50

modelManager = ModelManager('../data_processed/', dataset, overwrite=True)

featureExtractors = {'corr': utils.get_corr_features,
                     'pi_corr': utils.get_pers_img_corr_features,
                     'pl_corr': utils.get_pers_landscape_corr_features,
                     'pd_corr': utils.get_pers_diag_corr_features,
                     'pi': utils.get_pers_img_features,
                     'pl': utils.get_pers_landscape_features,
                     'pd': utils.get_pers_diag_features,
                     'pd_kern': utils.get_pers_diag_kern_features,
                     'pi_conv0': utils.get_pi_conv0_features,
                     'pi_conv1': utils.get_pi_conv1_features,
                     'pi_conv_dimchannel': utils.get_pi_conv_dimchannel_features,
                     'pi_conv_sum': utils.get_pi_conv_sum_features,
                     'pi_conv_hybrid': utils.get_pi_conv_hybrid_features
                    }

pi_conv_hybrid = NeuralNetClassifier(NNConvBranched(), max_epochs=EPOCHS, verbose=True)

modelManager.add_model(pi_conv_hybrid, 'pi_conv_hybrid', featureExtractors['pi_conv_hybrid'])
# modelManager.train('pi_conv_hybrid')

modelManager.train_all()
modelManager.evaluate_all(accuracy_score)
print(modelManager.tabulate())