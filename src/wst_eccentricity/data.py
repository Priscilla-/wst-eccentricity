"""Download the example dataset published on Zenodo.

The example dataset that accompanies the paper is archived on Zenodo under DOI
`10.5281/zenodo.21204498 <https://doi.org/10.5281/zenodo.21204498>`_
(CC-BY-4.0). This module downloads and unpacks it using only the Python
standard library (no extra dependencies).

The archive's file name and MD5 checksum are resolved at run time from the
Zenodo REST API, so the package keeps working if the archive is re-uploaded
or renamed on Zenodo.

From the command line::

    python -m wst_eccentricity.data --dest data

or programmatically::

    from wst_eccentricity.data import download_example_data
    root = download_example_data("data")   # -> path with waveforms/ and parameters/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
import zipfile
from pathlib import Path

__all__ = ["download_example_data", "find_dataset_root", "ZENODO_DOI", "ZENODO_RECORD"]

ZENODO_RECORD = "21204498"
ZENODO_DOI = "10.5281/zenodo.21204498"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

#: Junk produced by macOS archiving tools, skipped on extraction.
_MAC_JUNK_PREFIXES = ("__MACOSX/",)
_MAC_JUNK_BASENAMES = (".DS_Store",)


def _md5(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    """Return the hex MD5 digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _zenodo_zip_entry() -> dict:
    """Query the Zenodo API and return the record's ``.zip`` file entry.

    Returns
    -------
    dict
        The API entry of the dataset archive, with at least the keys
        ``"key"`` (file name), ``"checksum"`` (``"md5:<hex>"``) and
        ``"links"`` (holding the download URL).

    Raises
    ------
    RuntimeError
        If the record cannot be reached or holds no ``.zip`` file.
    """
    try:
        with urllib.request.urlopen(ZENODO_API, timeout=30) as resp:
            record = json.load(resp)
    except Exception as exc:  # noqa: BLE001 - report any network/parse failure
        raise RuntimeError(
            f"Could not query the Zenodo API at {ZENODO_API}: {exc}. "
            "Check your internet connection, or download the archive manually "
            f"from https://doi.org/{ZENODO_DOI} and unzip it yourself."
        ) from exc

    zips = [f for f in record.get("files", []) if f.get("key", "").endswith(".zip")]
    if not zips:
        raise RuntimeError(
            f"No .zip file found on Zenodo record {ZENODO_RECORD} "
            f"(https://doi.org/{ZENODO_DOI})."
        )
    # If several zips are ever published, take the largest (the dataset).
    return max(zips, key=lambda f: f.get("size", 0))


def _download(url: str, dest: str | os.PathLike, label: str) -> None:
    """Stream ``url`` to ``dest`` with a simple progress indicator."""
    def hook(block_num, block_size, total_size):
        if total_size > 0:
            done = min(block_num * block_size, total_size)
            pct = 100 * done / total_size
            print(f"\r  downloading {label}: {pct:5.1f}% "
                  f"({done / 1e6:.1f}/{total_size / 1e6:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print()


def _is_mac_junk(member: str) -> bool:
    """Return ``True`` for macOS metadata entries inside a zip archive."""
    name = member.replace("\\", "/")
    if any(name.startswith(p) or f"/{p}" in name for p in _MAC_JUNK_PREFIXES):
        return True
    base = os.path.basename(name.rstrip("/"))
    return base in _MAC_JUNK_BASENAMES or base.startswith("._")


def find_dataset_root(path: str | os.PathLike) -> Path:
    """Locate the directory that holds both ``waveforms/`` and ``parameters/``.

    Zip archives are sometimes wrapped in a top-level folder; this walks the
    extracted tree and returns the directory that contains the expected
    sub-folders. macOS ``__MACOSX`` metadata folders are ignored.

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
    candidates = [path] + sorted(
        p for p in path.rglob("*")
        if p.is_dir() and "__MACOSX" not in p.parts
    )
    for candidate in candidates:
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

    The archive name and checksum are looked up from the Zenodo API for record
    `21204498 <https://doi.org/10.5281/zenodo.21204498>`_, so this function
    keeps working if the published archive is renamed or replaced.

    Parameters
    ----------
    dest:
        Directory to download into and extract to (created if needed).
    extract:
        If ``True`` (default), unzip the archive after download (skipping
        macOS ``__MACOSX``/``.DS_Store`` metadata entries).
    check_md5:
        Verify the archive's MD5 checksum against the value published on
        Zenodo.
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

    entry = _zenodo_zip_entry()
    file_name = entry["key"]
    url = entry["links"].get("self") or entry["links"].get("download")
    expected_md5 = entry.get("checksum", "").removeprefix("md5:")
    zip_path = dest / file_name

    if not zip_path.exists():
        print(f"Fetching example dataset (DOI {ZENODO_DOI}) ...")
        _download(url, zip_path, label=file_name)
    else:
        print(f"Archive already present: {zip_path}")

    if check_md5 and expected_md5:
        print("  verifying checksum ...", end=" ", flush=True)
        digest = _md5(zip_path)
        if digest != expected_md5:
            raise ValueError(
                f"MD5 mismatch for {zip_path}: got {digest}, expected {expected_md5}. "
                "The download may be corrupted; delete the file and try again."
            )
        print("ok")

    if extract:
        print(f"  extracting into {dest} ...")
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if not _is_mac_junk(m)]
            zf.extractall(dest, members=members)
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
