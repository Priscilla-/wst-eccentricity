"""Sphinx configuration for wst-eccentricity."""

import os
import sys
from datetime import datetime

# Make the package importable for autodoc (src/ layout).
sys.path.insert(0, os.path.abspath("../src"))

project = "wst-eccentricity"
author = "Priscilla Canizares, Seppe J. Staelens, Isobel Romero-Shaw"
copyright = f"{datetime.now():%Y}, {author} (University of Cambridge)"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",          # NumPy / Google style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",                  # Markdown support
]

# Heavy / optional dependencies are mocked so docs build anywhere.
autodoc_mock_imports = ["torch", "kymatio", "h5py", "sklearn", "matplotlib", "numpy", "scipy"]
autodoc_typehints = "description"
autodoc_member_order = "bysource"

napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "wst-eccentricity"

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
