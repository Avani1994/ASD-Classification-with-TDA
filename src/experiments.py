from data_reader import ABIDEDataReader


def read_and_build_features():
    reader = ABIDEDataReader()
    subjects = reader.read('../data/ABIDE/rois_cc200', num_rois=200)
    [subject.compute_derived_measures() for subject in subjects]
    return subjects


read_and_build_features()
