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
        self.roi_time_series = roi_time_series
        self.correlation_matrix = None
        self.corr_vector = None
        self.persistence_diagram = None
        self.persistence_image = None
        self.persistence_landscape = None

    def compute_derived_measures(self, pd_metric_mapping=None):
        """Compute the Pearson correlation matrix (optionally, the persistence
        diagram as well from time series data

        :param pd_metric_mapping: specify the function to map correlations to a
            valid distance matrix. Default is sqrt(1 - max(0, x))
        """
        correlation = np.corrcoef(self.roi_time_series)

        self.corr_vector = correlation[np.tril_indices(correlation.shape[0], -1)]

        self.correlation_matrix = correlation

        if pd_metric_mapping is None:
            # TODO: try sqrt(2*(1-corr))
            def pd_metric_mapping(x): return np.sqrt(1 - np.clip(x, 0, None))
            # def pd_metric_mapping(x): return np.sqrt(0.5 * (1 - x))

        # Persistence diagram
        self.persistence_diagram       = Rips(maxdim=1, verbose=False).fit_transform(pd_metric_mapping(correlation),
                                                                                     distance_matrix=True)

        # Remove points at infinity
        self.persistence_diagram       = sklearn_tda.DiagramSelector().transform(self.persistence_diagram)

        # Vector representations of persistence diagrams
        self.persistence_image_list    = PersImage(pixels=(40, 40), verbose=False).transform(self.persistence_diagram)
        self.persistence_image         = sum(self.persistence_image_list).flatten()
        self.persistence_landscape     = sklearn_tda.Landscape().fit_transform(self.persistence_diagram).flatten()
