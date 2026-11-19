"""Reading NIfTI volumes: reorientation to RAS+ as in the original pipeline."""

import nibabel as nib
import numpy as np

from admri.common import load_nifti


def test_load_nifti_reorients_las_to_ras(tmp_path, volume):
    """FSL MNI152 templates are stored in LAS: x must be flipped, as Irene did."""
    las = np.diag([-1.0, 1.0, 1.0, 1.0])
    las[0, 3] = 90.0
    path = tmp_path / "las.nii.gz"
    nib.save(nib.Nifti1Image(volume, las), str(path))

    data, zooms = load_nifti(path)
    original = nib.as_closest_canonical(nib.load(str(path))).get_fdata()  # Irene's code
    np.testing.assert_array_equal(data, original.astype(np.float32))
    np.testing.assert_array_equal(data, volume[::-1, :, :])
    assert zooms == (1.0, 1.0, 1.0)

    raw, _ = load_nifti(path, canonical=False)
    np.testing.assert_array_equal(raw, volume)


def test_load_nifti_ras_unchanged(tmp_path, volume):
    path = tmp_path / "ras.nii.gz"
    nib.save(nib.Nifti1Image(volume, np.eye(4)), str(path))
    data, _ = load_nifti(path)
    np.testing.assert_array_equal(data, volume)
