# wst-eccentricity

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21204498.svg)](https://doi.org/10.5281/zenodo.21204498)

**Rapid classification of eccentric compact binaries with the wavelet scattering transform (WST).**

This repository accompanies the paper *"The Shape of Eccentricity: Rapid
Classification of Eccentric Binaries with the Wavelet Scattering Transform"*
(Canizares, Staelens & Romero-Shaw). It provides a compact, reproducible
pipeline that:

1. computes the **wavelet scattering transform** of gravitational-wave strain data, and
2. trains a lightweight **1D convolutional neural network** to distinguish
   eccentric from quasi-circular compact-binary signals.

> **Scope.** Waveform generation is intentionally *not* included. The pipeline
> starts from a **pre-generated waveform dataset and its parameter files**
> (see [Data layout](#data-layout)). Trained weights and datasets are
> distributed separately (see [Data & weights](#data--weights)).
>

## Authors

Priscilla Canizares, Seppe J. Staelens, and Isobel Romero-Shaw.


## Installation

```bash
git clone https://github.com/Priscilla-/wst-eccentricity.git
cd wst-eccentricity
pip install -e .            # add [plots] for plotting, [docs] for documentation
```

Python ≥ 3.10 is required. The main dependencies are PyTorch, Kymatio,
scikit-learn and h5py.

## Usage: download, then run

The whole pipeline is two commands.

**Step 1 — download the example dataset** (Zenodo DOI
[10.5281/zenodo.21204498](https://doi.org/10.5281/zenodo.21204498), CC-BY-4.0)
into a folder of your choice:

```bash
wst-eccentricity-download --dest data
# or: python -m wst_eccentricity.data --dest data
```

This fetches the dataset archive published on the record (its name and MD5
checksum are looked up from the Zenodo API), verifies the checksum, and
extracts the `waveforms/` and `parameters/` folders into `data/`.

**Step 2 — run the WST and the classifier of your choice:**

```bash
wst-eccentricity-run --data-dir data --classifier cnn1d --J 7 --Q 2
# or: python -m wst_eccentricity.pipeline --data-dir data --classifier cnn1d
```

This computes the wavelet scattering transform (caching it under
`data/transform_coefficients/`), builds the labelled dataset, trains the chosen
classifier, and prints the AUC, average precision and the true-positive rate at
a target false-positive rate.

Built-in classifiers: `cnn1d` (the paper's 1D-CNN, default) and `logreg` (a
logistic-regression baseline). See [Choosing / adding a classifier](#choosing--adding-a-classifier).

### Programmatic use

```python
from wst_eccentricity import run_pipeline

results = run_pipeline("data", J=7, Q=2, classifier="cnn1d", target_fpr=0.1)
print(results["auc"], results["tpr"])
```

Lower-level building blocks are also exposed, e.g.:

```python
from wst_eccentricity import compute_scattering, build_dataset, SWT_CNN_1D_Binned, train_binary

# gws: torch.Tensor of shape (N, D, T) -- N signals, D detectors, T samples
Sx, meta = compute_scattering(gws, J=7, Q=2)   # (N, D, C, T); fed directly to SWT_CNN_1D_Binned
```

### No-data smoke test

To exercise the training/evaluation path without any dataset:

```bash
python examples/quickstart.py --demo
```

## Choosing / adding a classifier

`--classifier` selects an entry from `wst_eccentricity.CLASSIFIERS`. To plug in
your own model, register a callable that returns validation and test
probabilities:

```python
from wst_eccentricity import register_classifier, run_pipeline

@register_classifier("my_model")
def my_model(X_train, y_train, X_val, y_val, X_test, device, **kw):
    ...  # train, then:
    return val_probs, test_probs

run_pipeline("data", classifier="my_model")
```

## Data layout

The pipeline expects a directory of the form:

```
dataset/
├── waveforms/                # one <name>.hdf5 per signal (single array each)
├── parameters/               # one params_*.txt per signal ("key: value" lines)
└── transform_coefficients/   # WST_{J}_{Q}_*.pt written by the pipeline
```

Waveform files and parameter files are matched by the `s<seed>_<index>` tag in
their names. Each parameter file must contain at least an `eccentricity:` line;
the binary label is `eccentricity > e_thr` (default `e_thr = 0.01`).

## Package overview

| Module | Contents |
| --- | --- |
| `wst_eccentricity.data` | download the Zenodo example dataset |
| `wst_eccentricity.io` | load waveform HDF5 files, parameter files, WST tensors |
| `wst_eccentricity.transforms` | compute the WST (single batch or streamed to disk) |
| `wst_eccentricity.datasets` | build labelled datasets, HDF5 cache, `Dataset` class |
| `wst_eccentricity.models` | `SWT_CNN_1D_Binned` reference classifier |
| `wst_eccentricity.training` | training loop with early stopping |
| `wst_eccentricity.metrics` | FPR/TPR, ROC threshold selection, AUC/AP |
| `wst_eccentricity.pipeline` | end-to-end `run_pipeline` + classifier registry |
| `wst_eccentricity.plotting` | ROC, confusion matrix, loss history (optional) |

## Data & weights

The waveform datasets and trained network weights are **not** stored in this
repository. The example dataset is archived on Zenodo
([10.5281/zenodo.21204498](https://doi.org/10.5281/zenodo.21204498)) and can be
fetched with `python -m wst_eccentricity.data` (see
[Example dataset](#example-dataset) above).

## Citing

If you use this code, please cite the paper and the software (see
[`CITATION.cff`](CITATION.cff)).

## License

Released under the [MIT License](LICENSE).


