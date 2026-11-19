import numpy as np
import pytest

from admri.slices import extract_patches, extract_slices, normalize_volume, slice_indices


def test_slice_indices_odd_and_even():
    assert slice_indices((182, 218, 182), "coronal", 3) == [108, 109, 110]
    idx = slice_indices((182, 218, 182), "coronal", 20)
    assert idx[0] == 99 and idx[-1] == 118 and len(idx) == 20
    assert slice_indices((182, 218, 182), "axial", 1) == [91]


def test_slice_indices_out_of_bounds():
    with pytest.raises(ValueError):
        slice_indices((10, 10, 10), "coronal", 3, center=0)


def test_extract_slices_shapes(volume):
    pairs = extract_slices(volume, "coronal", 3)
    assert [i for i, _ in pairs] == [108, 109, 110]
    assert pairs[0][1].shape == (182, 182)
    assert extract_slices(volume, "axial", 1)[0][1].shape == (218, 182)


def test_normalize_volume_range(volume):
    out = normalize_volume(volume)
    assert out.min() == 0 and out.max() == pytest.approx(1)
    assert np.all(out[volume <= 0] == 0)


def test_patches_mirror():
    img = np.arange(182 * 182, dtype=np.float32).reshape(182, 182)
    p = extract_patches(img)
    assert p["right"].shape == (64, 64) and p["leftflip"].shape == (64, 64)
    np.testing.assert_array_equal(p["leftflip"], img[90:154, 12:76][:, ::-1])
    np.testing.assert_array_equal(p["right"], img[90:154, 106:170])
