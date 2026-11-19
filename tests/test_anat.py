"""Anatomical preprocessing steps with a fake ``fun_defs`` module (no HD-BET/ANTs needed)."""

import shutil
import sys

import pytest
import yaml

from admri import anat
from admri.bids import load_layout, select_image, source_entities

FAKE_FUN_DEFS = '''
import shutil, os
def skull_stripping_hd_bet(path_in, name_in, path_out):
    out = os.path.join(path_out, name_in.replace(".nii.gz", "_seg-brain-hdbet.nii.gz"))
    shutil.copy(os.path.join(path_in, name_in), out)
    return out
def ants_registration_subject_T1_2_template(name_template, name_mov_t1, name_t1_reg):
    shutil.copy(name_mov_t1, name_t1_reg)
    prefix = name_t1_reg[:-len(".nii.gz")]
    shutil.copy(name_mov_t1, prefix + "_transform0.nii.gz")
    open(prefix + "_transform1.mat", "w").write("affine")
'''


@pytest.fixture
def fake_tools(tmp_path):
    folder = tmp_path / "tools"
    folder.mkdir()
    (folder / "fun_defs.py").write_text(FAKE_FUN_DEFS)
    yield folder
    sys.modules.pop("fun_defs", None)
    if str(folder) in sys.path:
        sys.path.remove(str(folder))


def test_preprocess_anat_subject_bids_names(tmp_path, bids_dataset, fake_tools):
    root, _ = bids_dataset
    deriv = tmp_path / "deriv"
    f = select_image(load_layout(root), "02", "01", prefer={"run": [2]})
    ents = source_entities(f)
    config = {
        "derivatives": {"space": "MNI152NLin6Asym"},
        "group_tools": {"path": str(fake_tools), "bin_path": None},
        "skull_strip": {"method": "hd-bet"},
        "registration": {"template": "tpl.nii.gz"},
        "tissue_maps": {"enabled": False},
    }
    out = anat.preprocess_anat_subject(f.path, ents, deriv, config)
    for key in ("brain_native", "brain_template", "warp", "affine"):
        assert out[key].is_file(), key
    assert out["brain_template"].name == (
        "sub-02_ses-01_run-02_space-MNI152NLin6Asym_desc-brain_T1w.nii.gz")
    assert out["warp"].name == (
        "sub-02_ses-01_run-02_from-T1w_to-MNI152NLin6Asym_desc-warp_xfm.nii.gz")
    assert not (out["warp"].parent / "_work_skullstrip").exists()
    assert not (out["warp"].parent / "_work_registration").exists()


def test_missing_group_tools(tmp_path):
    with pytest.raises(FileNotFoundError):
        anat.load_group_tools(tmp_path)


def test_steps_can_run_separately(tmp_path, bids_dataset, fake_tools):
    """Each step works on its own and is skipped when its outputs exist."""
    root, _ = bids_dataset
    deriv = tmp_path / "deriv"
    f = select_image(load_layout(root), "01", "01")
    config = {"skull_strip": {"method": "hd-bet"}, "registration": {"template": "tpl.nii.gz"}}
    tools = anat.load_group_tools(fake_tools)
    outputs = anat.anat_outputs(deriv, source_entities(f))

    # Registration alone fails: there is no skull-stripped brain yet.
    with pytest.raises(FileNotFoundError):
        anat.register_brain_subject(outputs, config, tools)

    anat.skull_strip_subject(f.path, outputs, config, tools)
    assert outputs["brain_native"].is_file()
    assert not outputs["brain_template"].exists()
    assert not (outputs["brain_native"].parent / "_work_skullstrip").exists()

    anat.register_brain_subject(outputs, config, tools)
    for key in ("brain_template", "warp", "affine"):
        assert outputs[key].is_file(), key
    assert not (outputs["brain_native"].parent / "_work_registration").exists()

    # Second call without overwrite: outputs exist, nothing is recomputed.
    outputs["affine"].write_text("unchanged")
    anat.register_brain_subject(outputs, config, tools)
    assert outputs["affine"].read_text() == "unchanged"


def test_run_anat_saves_log_and_config(tmp_path, bids_dataset, fake_tools):
    """run_anat writes the log and the configuration actually used."""
    from admri.common import load_config
    root, _ = bids_dataset
    deriv = tmp_path / "deriv"
    exp = tmp_path / "anat.yaml"
    exp.write_text(yaml.safe_dump({
        "bids_root": str(root), "derivatives_root": str(deriv),
        "bids": {"prefer": {"run": [2]}},
        "group_tools": {"path": str(fake_tools)},
        "registration": {"template": "tpl.nii.gz"},
    }))
    log = anat.run_anat(load_config(exp), start=0, end=2, n_jobs=1, overwrite=True)
    assert (log["status"] == "ok").all() and len(log) == 2
    assert (deriv / "logs" / "anat_0-2.csv").is_file()
    used = yaml.safe_load((deriv / "logs" / "anat_config_used_0-2.yaml").read_text())
    assert used["skull_strip"]["method"] == "hd-bet"
    assert used["run"] == {"n_jobs": 1, "overwrite": True}
