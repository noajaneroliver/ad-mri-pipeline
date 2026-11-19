"""Shared fixtures: a synthetic brain-like volume and a tiny BIDS dataset."""

import json

import nibabel as nib
import numpy as np
import pytest


def synthetic_volume(shape=(182, 218, 182), seed=0):
    """Ellipsoid 'brain' with smooth intensity variations and noise; background 0."""
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
    inside = (x / 0.8) ** 2 + (y / 0.85) ** 2 + (z / 0.75) ** 2 < 1
    vol = 300 + 200 * np.sin(4 * x) * np.cos(3 * y) + 100 * z + rng.normal(0, 30, shape)
    vol[~inside] = 0
    return vol.astype(np.float32)


@pytest.fixture
def volume():
    return synthetic_volume()


def _save(vol, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(path))


@pytest.fixture
def bids_dataset(tmp_path):
    """
    BIDS dataset with 3 subjects; sub-02 has run-01 and run-02, sub-03 has no session.
    Registered images are written directly as derivatives (registration is skipped).
    """
    root = tmp_path / "bids"
    root.mkdir()
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "test", "BIDSVersion": "1.9.0"}))
    small = np.zeros((4, 4, 4), np.float32)
    _save(small, root / "sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz")
    _save(small, root / "sub-02/ses-01/anat/sub-02_ses-01_run-01_T1w.nii.gz")
    _save(small, root / "sub-02/ses-01/anat/sub-02_ses-01_run-02_T1w.nii.gz")
    _save(small, root / "sub-03/anat/sub-03_T1w.nii.gz")

    deriv = tmp_path / "derivatives" / "admri"
    vol = synthetic_volume()
    for name, folder in [("sub-01_ses-01", "sub-01/ses-01"),
                         ("sub-02_ses-01_run-02", "sub-02/ses-01"),
                         ("sub-03", "sub-03")]:
        _save(vol, deriv / folder / "anat" / f"{name}_space-MNI152NLin6Asym_desc-brain_T1w.nii.gz")
        for label in ["CSF", "GM", "WM"]:
            _save(np.clip(vol / vol.max(), 0, 1),
                  deriv / folder / "anat" / f"{name}_space-MNI152NLin6Asym_label-{label}_probseg.nii.gz")
    return root, deriv
