from ripser import Rips
from persim import PersImage
import sklearn_tda
import numpy as np


class Subject:
    """Class to represent a subject of the ABIDE dataset. Contains information
    about the subject ID, autism label and time series. Allows computing
    derived measures, namely correlations and persistence diagrams from the
    time series data.

    :param subject_id: subject ID of the individual
    :param label: [0|1] whether the individual is autistic (0) or control (1)
    :param roi_time_series: per ROI time series data
    """

    def __init__(self, subject_id, label, roi_time_series):
        self.subject_id = subject_id
        self.label = label
        self.rois_time_series = roi_time_series
        self.correlation_matrix = None
        self.persistence_diagram = None
        self.persistence_image = None
        self.persistence_landscape = None
        self.scale_space_kernel = None
        self.weighted_gaussian_kernel = None
        self.sliced_wasserstein_kernel = None
        self.fisher_kernel = None

    def compute_derived_measures(self, pd_metric_mapping=None):
        """Compute the Pearson correlation matrix (optionally, the persistence
        diagram as well from time series data

        :param pd_metric_mapping: specify the function to map correlations to a
            valid distance matrix. Default is sqrt(1 - max(0, x))
        """
        correlation = np.corrcoef(self.rois_time_series)

        if pd_metric_mapping is None:
            # TODO: try sqrt(2*(1-corr))
            def pd_metric_mapping(x): return np.sqrt(1 - np.clip(x, 0, None))
            # def pd_metric_mapping(x): return np.sqrt(2 * (1 - x))

        # Persistence diagram
        self.persistence_diagram       = Rips(maxdim=1, verbose=False).fit_transform(pd_metric_mapping(correlation),
                                                                      distance_matrix=True)

        # Remove points at infinity
        self.persistence_diagram       = sklearn_tda.DiagramSelector().transform(self.persistence_diagram)

        # Vector representations of persistence diagrams
        self.persistence_image         = PersImage(spread=1, pixels=correlation.shape, verbose=False).transform(self.persistence_diagram)
        self.persistence_landscape     = sklearn_tda.Landscape().fit_transform(self.persistence_diagram)

        # Kernels over persistence diagrams

        def kernel(K, diagrams):
            return [K.fit(diagram) for diagram in diagrams]

        # self.scale_space_kernel        = sklearn_tda.PersistenceScaleSpaceKernel().fit_transform(self.persistence_diagram[0])
        # self.scale_space_kernel        = sklearn_tda.PersistenceScaleSpaceKernel().fit_transform(self.persistence_diagram)
        self.weighted_gaussian_kernel  = sklearn_tda.PersistenceWeightedGaussianKernel().fit_transform(self.persistence_diagram)
        self.sliced_wasserstein_kernel = sklearn_tda.SlicedWassersteinKernel().fit_transform(self.persistence_diagram)
        self.fisher_kernel             = sklearn_tda.PersistenceFisherKernel().fit_transform(self.persistence_diagram)

        # UMAP based dimensionality reduction
        # TODO - UMAP dimensionality reduction
