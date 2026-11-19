"""
Check that the new code reproduces the original pipeline (Irene's TFM).

The ``legacy_*`` functions below are copied from the original scripts
(extract_*_slices.py, image_resize_def.py, gaussian_normalization_def.py,
extract_patches_inferiors.py), keeping their PNG round-trips. The new pipeline
with ``legacy_uint8: true`` must give exactly the same 8-bit images.
"""

import numpy as np
import pytest
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter

from admri.common import DEFAULTS, to_uint8
from admri.slices import extract_patches, extract_slices, normalize_volume, preprocess_slice

SIGMA = 6.0 / 2.355


# --- original code --------------------------------------------------------

def legacy_save_png(slice_2d, out_path):
    slice_2d = np.clip(slice_2d, 0.0, 1.0)
    arr = (slice_2d * 255).round().astype(np.uint8)
    Image.fromarray(arr, mode="L").save(out_path)


def legacy_resize_direct(in_path, out_path, size=(224, 224)):
    with Image.open(in_path) as img:
        img.convert("L").resize(size, Image.Resampling.LANCZOS).save(out_path)


def legacy_resize_crop(in_path, out_path, target=(224, 224)):
    with Image.open(in_path) as img:
        img = img.convert("L")
        tw, th = target
        w, h = img.size
        scale = max(tw / w, th / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        r = img.resize((nw, nh), Image.Resampling.LANCZOS)
        left, top = (nw - tw) // 2, (nh - th) // 2
        r.crop((left, top, left + tw, top + th)).save(out_path)


def legacy_gaussian(in_path, out_path):
    with Image.open(in_path) as img:
        original = np.array(img.convert("L")).astype(np.float32) / 255.0
    mask = original > 0.0
    smooth = gaussian_filter(original, sigma=SIGMA)
    vals = smooth[mask]
    final = np.clip((original - vals.min()) / (vals.max() - vals.min()), 0, 1)
    Image.fromarray((final * 255).round().astype(np.uint8)).save(out_path)


def legacy_slice(vol, view, idx):
    data = normalize_volume(vol, 0.0)  # identical to the original normalize_volume
    sl = data[:, idx, :] if view == "coronal" else data[:, :, idx]
    return np.nan_to_num(np.rot90(sl), nan=0.0)


# --- helpers ---------------------------------------------------------------

def new_config(**overrides):
    sl = {**DEFAULTS["slices"], "legacy_uint8": True}
    for k, v in overrides.items():
        sl[k] = {**sl[k], **v} if isinstance(v, dict) and isinstance(sl.get(k), dict) else v
    return sl


def new_pipeline(volume, sl):
    vol = normalize_volume(volume, sl["background_threshold"])
    return {i: to_uint8(preprocess_slice(s, sl)) for i, s in extract_slices(vol, sl["view"], sl["n_slices"], sl["center"], sl["rotate90"])}


def read(path):
    with Image.open(path) as img:
        return np.array(img)


# --- tests -----------------------------------------------------------------

@pytest.mark.parametrize("view,method,legacy_resize", [
    ("coronal", "direct", legacy_resize_direct),
    ("axial", "crop_preserve_aspect", legacy_resize_crop),
])
def test_slices_match_original(volume, tmp_path, view, method, legacy_resize):
    sl = new_config(view=view, n_slices=3, resize={"size": 224, "method": method})
    new = new_pipeline(volume, sl)

    for idx, new_img in new.items():
        a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
        legacy_save_png(legacy_slice(volume, view, idx), a)
        legacy_resize(a, b)
        legacy_gaussian(b, c)
        old_img = read(c)
        assert old_img.shape == new_img.shape == (224, 224)
        np.testing.assert_array_equal(new_img, old_img)


def test_patches_match_original(volume, tmp_path):
    sl = new_config(view="coronal", n_slices=3, resize={"method": None},
                    patches={"patch_size": 64, "margin_bottom": 28, "margin_side": 12})
    vol = normalize_volume(volume, 0.0)
    for idx, image in extract_slices(vol, sl["view"], sl["n_slices"], sl["center"], sl["rotate90"]):
        new = {k: to_uint8(preprocess_slice(p, sl)) for k, p in
               extract_patches(image, **sl["patches"]).items()}

        a = tmp_path / "slice.png"
        legacy_save_png(legacy_slice(volume, "coronal", idx), a)
        img = Image.open(a)
        w, h = img.size
        y1, y2 = h - 28 - 64, h - 28
        right = img.crop((w - 12 - 64, y1, w - 12, y2))
        left = ImageOps.mirror(img.crop((12, y1, 76, y2)))
        for key, patch in [("right", right), ("leftflip", left)]:
            p_in, p_out = tmp_path / f"{key}.png", tmp_path / f"{key}_n.png"
            patch.save(p_in)
            legacy_gaussian(p_in, p_out)
            np.testing.assert_array_equal(new[key], read(p_out))


def test_float_pipeline_close_to_original(volume):
    """Without legacy rounding the result differs only by small amounts."""
    sl_new = new_config(legacy_uint8=False, resize={"method": "direct"})
    sl_old = new_config(resize={"method": "direct"})
    vol = normalize_volume(volume, 0.0)
    a = {i: preprocess_slice(s, sl_new) for i, s in extract_slices(vol, sl_new["view"], sl_new["n_slices"], sl_new["center"], sl_new["rotate90"])}
    b = {i: preprocess_slice(s, sl_old) for i, s in extract_slices(vol, sl_old["view"], sl_old["n_slices"], sl_old["center"], sl_old["rotate90"])}
    for i in a:
        assert np.abs(a[i] - b[i]).mean() < 2 / 255
