"""End-to-end test of the ``slices`` step on a tiny BIDS dataset."""

import pandas as pd
import yaml

from admri.bids import list_images
from admri.common import load_config, load_image
from admri.slices import run_slices


def write_configs(tmp_path, root, deriv, slices):
    ds = tmp_path / "dataset.yaml"
    ds.write_text(yaml.safe_dump({
        "bids_root": str(root), "derivatives_root": str(deriv),
        "bids": {"prefer": {"run": [2]}, "subject_list": str(tmp_path / "subjects.csv")},    }))
    exp = tmp_path / "exp.yaml"
    exp.write_text(yaml.safe_dump({"dataset": str(ds), "name": "test", "slices": slices}))
    return load_config(exp)


def test_list_images(tmp_path, bids_dataset):
    root, deriv = bids_dataset
    config = write_configs(tmp_path, root, deriv, {})
    images = list_images(config)
    assert len(images) == 3
    assert images[1]["entities"] == {"subject": "02", "session": "01", "run": 2}


def test_run_slices_t1(tmp_path, bids_dataset):
    root, deriv = bids_dataset
    config = write_configs(tmp_path, root, deriv, {"view": "coronal", "n_slices": 3})
    manifest = run_slices(config)

    assert len(manifest) == 9
    assert sorted(manifest["slice_position"].unique()) == [-1, 0, 1]
    img = load_image(manifest["path"].iloc[0])
    assert img.shape == (224, 224) and 0 <= img.min() and img.max() <= 1
    out = deriv / "slices" / "test"
    assert (out / "config_used.yaml").is_file()
    log = pd.read_csv(out / "log.csv")
    assert (log["status"] == "ok").all()
    assert "run-02" in manifest.loc[manifest["subject"] == "02", "path"].iloc[0]


def test_run_slices_tissue_maps(tmp_path, bids_dataset):
    root, deriv = bids_dataset
    config = write_configs(tmp_path, root, deriv, {
        "source": "tissue_maps", "n_slices": 1, "volume_normalization": None,
        "resize": {"method": "direct"}, "normalization": {"method": None},
    })
    manifest = run_slices(config)
    assert sorted(manifest["label"].unique()) == ["CSF", "GM", "WM"]
    assert len(manifest) == 9


def test_run_slices_patches_parallel(tmp_path, bids_dataset):
    root, deriv = bids_dataset
    config = write_configs(tmp_path, root, deriv, {
        "n_slices": 3, "resize": {"method": None},
        "patches": {"patch_size": 64, "margin_bottom": 28, "margin_side": 12},
        "output_format": "png",
    })
    manifest = run_slices(config, n_jobs=2)
    assert len(manifest) == 3 * 3 * 2
    assert load_image(manifest["path"].iloc[0]).shape == (64, 64)
