# Autism Classification with TDA

### Directory Structure
---
* ``data_reader.py`` - Contains structure to read the ABIDE dataset and compute the required 
    topological features from the data
* ``features.py`` - Contains the `Subject` class that holds the data and computed features for 
    each individual in the ABIDE data. The computed features are - persistence diagram, 
    persistence image and persistence landscape
* ``models.py`` - Implements various models used in the experiments
* ``permutation_test.py`` - Implements the one-sided permutation test for given performance metric
* ``slayer.py`` - code from [CHofer](https://github.com/c-hofer/chofer_torchex) that implements 
    the persistent homology based layer from this paper: *Hofer, Christoph, et al. "Deep learning with 
    topological signatures." Advances in Neural Information Processing Systems. 2017.*
* ``utils.py`` - contains utility functions for loading data and extracting model specific features.

#### TODOs
- [] Docstrings for all functions
- [] Data download script