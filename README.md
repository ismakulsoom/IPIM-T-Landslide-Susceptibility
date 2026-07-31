
# IPIM-T: InSAR-Constrained Physics-Informed Multi-Stream Transformer

Official implementation for landslide susceptibility assessment.

## Overview

IPIM-T integrates:

- multi-stream feature encoding
- InSAR deformation representation
- cross-attention fusion
- physics-informed regularization


## Dataset

Demo data:

data/demo/

- X_env_demo.npy
- X_insar_demo.npy
- y_demo.npy


The complete KKH dataset is not included because of data volume and data-sharing restrictions.


## Installation

```bash
pip install -r requirements.txt
```


## Quick test

```bash
python examples/quick_test.py
```


## Training

```bash
python scripts/train.py
```


## Evaluation

```bash
python scripts/evaluate.py
```


## Citation

Please cite the IPIM-T paper.
