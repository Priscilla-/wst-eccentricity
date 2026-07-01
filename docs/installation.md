# Installation

```bash
git clone https://github.com/Priscilla-/wst-eccentricity.git
cd wst-eccentricity
pip install -e .
```

Optional extras:

```bash
pip install -e ".[plots]"   # matplotlib helpers
pip install -e ".[docs]"    # build this documentation
pip install -e ".[dev]"     # tests + plots + docs
```

Python ≥ 3.10 is required. Core dependencies: PyTorch, Kymatio, scikit-learn,
h5py, NumPy, SciPy.

A conda environment file is also provided:

```bash
conda env create -f environment.yml
conda activate wst-eccentricity
```
