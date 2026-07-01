"""Download the example dataset published on Zenodo.

The example dataset that accompanies the paper is archived on Zenodo under DOI
`10.5281/zenodo.21108640 <https://doi.org/10.5281/zenodo.21108640>`_
(CC-BY-4.0). This module downloads and unpacks it using only the Python
standard library (no extra dependencies).

From the command line::

    python -m wst_eccentricity.data --dest data

or programmatically::

    from wst_eccentricity.data import download_example_data
    root = download_example_data("data")   # -> path with waveforms/ and parameters/
"""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path

__all__ = ["download_example_data", "find_dataset_root", "ZENODO_DOI", "EXAMPLE_FILE"]

ZENODO_RECORD = "21108640"
ZENODO_DOI = "10.5281/zenodo.21108640"
EXAMPLE_FILE = "data_2026-04-27.zip"
EXAMPLE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{EXAMPLE_FILE}?download=1"
EXAMPLE_MD5 = "6a4cae064ea8363d2e4aeca73a77e22d"


def _md5(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    """Return the hex MD5 digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: str | os.PathLike) -> None:
    """Stream ``url`` to ``dest`` with a simple progress indicator."""
    def hook(block_num, block_size, total_size):
        if total_size > 0:
            done = min(block_num * block_size, total_size)
            pct = 100 * done / total_size
            print(f"\r  downloading {EXAMPLE_FILE}: {pct:5.1f}% "
                  f"({done / 1e6:.1f}/{total_size / 1e6:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print()


def find_dataset_root(path: str | os.PathLike) -> Path:
    """Locate the directory that holds both ``waveforms/`` and ``parameters/``.

    Zip archives are sometimes wrapped in a top-level folder; this walks the
    extracted tree and returns the directory that contains the expected
    sub-folders.

    Parameters
    ----------
    path:
        Directory to search.

    Returns
    -------
    pathlib.Path
        The dataset root. Falls back to ``path`` itself if no match is found.
    """
    path = Path(path)
    for candidate in [path, *sorted(p for p in path.rglob("*") if p.is_dir())]:
        if (candidate / "waveforms").is_dir() and (candidate / "parameters").is_dir():
            return candidate
    return path


def download_example_data(
    dest: str | os.PathLike = "data",
    extract: bool = True,
    check_md5: bool = True,
    keep_zip: bool = False,
) -> Path:
    """Download (and optionally unpack) the Zenodo example dataset.

    Parameters
    ----------
    dest:
        Directory to download into and extract to (created if needed).
    extract:
        If ``True`` (default), unzip the archive after download.
    check_md5:
        Verify the archive's MD5 checksum against the published value.
    keep_zip:
        Keep the downloaded ``.zip`` after extraction (default: remove it).

    Returns
    -------
    pathlib.Path
        Path to the dataset root (the folder containing ``waveforms/`` and
        ``parameters/``) if it can be located, otherwise ``dest``.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / EXAMPLE_FILE

    if not zip_path.exists():
        print(f"Fetching example dataset (DOI {ZENODO_DOI}) ...")
        _download(EXAMPLE_URL, zip_path)
    else:
        print(f"Archive already present: {zip_path}")

    if check_md5:
        print("  verifying checksum ...", end=" ", flush=True)
        digest = _md5(zip_path)
        if digest != EXAMPLE_MD5:
            raise ValueError(
                f"MD5 mismatch for {zip_path}: got {digest}, expected {EXAMPLE_MD5}. "
                "The download may be corrupted; delete the file and try again."
            )
        print("ok")

    if extract:
        print(f"  extracting into {dest} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        if not keep_zip:
            zip_path.unlink()

    root = find_dataset_root(dest)
    print(f"Done. Dataset root: {root}")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Zenodo example dataset.")
    parser.add_argument("--dest", default="data", help="target directory (default: data)")
    parser.add_argument("--no-extract", action="store_true", help="download only, do not unzip")
    parser.add_argument("--no-md5", action="store_true", help="skip checksum verification")
    parser.add_argument("--keep-zip", action="store_true", help="keep the .zip after extraction")
    args = parser.parse_args()
    download_example_data(
        dest=args.dest,
        extract=not args.no_extract,
        check_md5=not args.no_md5,
        keep_zip=args.keep_zip,
    )


if __name__ == "__main__":
    main()
