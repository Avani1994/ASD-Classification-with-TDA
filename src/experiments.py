from data_reader import ABIDEDataReader
import pickle
import sklearn_tda


NUM_ROIS = 200


def read_and_build_features(fresh=False):
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
            subjects = reader.read(raw_data_location, num_rois=NUM_ROIS)

            with open(dump_location, 'wb') as f:
                pickle.dump(subjects, f)

        return subjects

    else:
        reader = ABIDEDataReader()
        subjects = reader.read(raw_data_location, num_rois=NUM_ROIS)

        with open(dump_location, 'wb') as f:
            pickle.dump(subjects, f)

        return subjects


data = read_and_build_features()

