# Autism Classification with TDA

This repository contains the code for the paper [Autism Classification Using Topological Features
and Deep Learning: A Cautionary Tale](https://link.springer.com/chapter/10.1007/978-3-030-32248-9_82).

The driver code can be found in `src/Experiments.ipynb` - it contains the code to load the 
dataset, extract features, define and evalaute models.

<hr>

### Requirements
The code is written for Python 3.7 and has been tested with the 
following package configuration. Older versions of the package 
may work but I haven't tested those.
```
numpy==1.18.5
tqdm==4.47.0
sklearn==0.23.1
skorch==0.8.0
ripser==0.3.2
sklearn_tda               # https://github.com/MathieuCarriere/sklearn-tda
torch==1.2.0
pandas==0.25.1
scipy==1.5.0
statsmodels==0.11.1
```

<hr>

If you find this code useful, please cite our paper:

```
@inproceedings{RathorePalandeWang2019,
  title={Autism Classification Using Topological Features and Deep Learning: A Cautionary Tale},
  author={Rathore, Archit and Palande, Sourabh and Anderson, Jeffrey S and Zielinski, Brandon A and Fletcher, P Thomas and Wang, Bei},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={736--744},
  year={2019},
  organization={Springer}
}
```
