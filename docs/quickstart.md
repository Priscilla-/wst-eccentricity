# Quickstart

The pipeline is two commands: **download**, then **run**.

## Step 1 — download the example dataset

Published on Zenodo (DOI
[10.5281/zenodo.21108640](https://doi.org/10.5281/zenodo.21108640), CC-BY-4.0):

```bash
wst-eccentricity-download --dest data
```

This downloads `data_2026-04-27.zip`, verifies its checksum, and extracts the
`waveforms/` and `parameters/` folders into `data/`.

## Step 2 — run WST + classifier

```bash
wst-eccentricity-run --data-dir data --classifier cnn1d --J 7 --Q 2
```

Built-in classifiers: `cnn1d` (default) and `logreg`. The command computes the
WST (cached under `data/transform_coefficients/`), builds the labelled dataset,
trains the classifier, and prints AUC, average precision and the TPR at the
target FPR.

## Programmatic use

```python
from wst_eccentricity import run_pipeline

results = run_pipeline("data", J=7, Q=2, classifier="cnn1d", target_fpr=0.1)
print(results["auc"], results["tpr"])
```

Lower-level building blocks:

```python
import torch
from wst_eccentricity import compute_scattering, build_dataset, standardize, SWT_CNN_1D_Binned

gws = torch.randn(64, 3, 4096)                      # (N signals, D detectors, T samples)
Sx, meta = compute_scattering(gws, J=7, Q=2)        # (N, D, C, T)
# Sx, y, params = build_dataset("data/parameters", "data/transform_coefficients", J=7, Q=2)
X = standardize(Sx)                                 # keep the native 4D shape
model = SWT_CNN_1D_Binned(in_channels=X.shape[2], num_detectors=X.shape[1])
```

## No-data smoke test

```bash
python examples/quickstart.py --demo
```

## Adding a classifier

```python
from wst_eccentricity import register_classifier

@register_classifier("my_model")
def my_model(X_train, y_train, X_val, y_val, X_test, device, **kw):
    ...
    return val_probs, test_probs
```

(data-layout)=
## Data layout

```
dataset/
├── waveforms/                # one <name>.hdf5 per signal
├── parameters/               # one params_*.txt per signal ("key: value" lines)
└── transform_coefficients/   # WST_{J}_{Q}_*.pt written by the pipeline
```

Each parameter file must contain an `eccentricity:` entry; the binary label is
`eccentricity > e_thr` (default `e_thr = 0.01`).
