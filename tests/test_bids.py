from admri import bids as b


def test_build_bids_name_order():
    name = b.build_bids_name(
        {"description": "brain", "space": "MNI152NLin6Asym", "run": 2, "subject": "01", "session": "01"},
        "T1w", ".nii.gz")
    assert name == "sub-01_ses-01_run-02_space-MNI152NLin6Asym_desc-brain_T1w.nii.gz"


def test_select_prefers_run02(bids_dataset):
    root, _ = bids_dataset
    layout = b.load_layout(root)
    assert b.get_subjects(layout) == ["01", "02", "03"]
    f = b.select_image(layout, "02", "01", prefer={"run": [2]})
    assert f.path.endswith("run-02_T1w.nii.gz")
    f = b.select_image(layout, "02", "01", prefer={"run": ["01"]})
    assert f.path.endswith("run-01_T1w.nii.gz")


def test_sessions(bids_dataset):
    root, _ = bids_dataset
    layout = b.load_layout(root)
    assert b.get_sessions(layout, "01") == ["01"]
    assert b.get_sessions(layout, "01", "ses-02") == []
