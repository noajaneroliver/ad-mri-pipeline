"""
Tools shared by the two preprocessing steps.

- **Configuration**: load, merge and validate the YAML files (:data:`DEFAULTS`, :func:`load_config`).
- **Reading and writing images**: NIfTI volumes and the saved 2D images.
- **Running a step for many images**: one job per image, sequentially or in parallel, with a CSV log.
"""

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Default values used when a key is missing from the experiment config.
DEFAULTS = {
    "bids": {
        "subjects": None,
        "subject_list": None,
        "datatype": "anat",
        "suffix": "T1w",
        "extension": ".nii.gz",
        "filters": {},
        "prefer": {},
        "database_path": None,
    },
    "derivatives": {
        "pipeline_name": "admri",
        "space": "MNI152NLin6Asym",
    },
    "group_tools": {
        "path": None,
        "bin_path": None,
    },
    "skull_strip": {
        "method": "hd-bet",
        "device": "cuda"      # cuda (GPU) or cpu
    },
    "registration": {
        "template": None,
    },
    "tissue_maps": {
        "enabled": False,
        "segmentation_dir": None,
        "pattern": None,
        "interpolation": "linear",
    },
    "slices": {
        "source": "T1w",
        "view": "coronal",
        "n_slices": 3,
        "center": None,
        "rotate90": True,
        "volume_normalization": "minmax",
        "background_threshold": 0.0,
        "patches": None,
        "resize": {"size": 224, "method": "crop_preserve_aspect"},
        "normalization": {"method": "gaussian", "fwhm": 6.0, "pixel_size": 1.0},
        "legacy_uint8": False,
        "output_format": "npy",
    },
    "run": {
        "n_jobs": 1,
        "overwrite": False,
    },
}

VALID_VIEWS = {"axial", "coronal", "sagittal"}

VALID_RESIZE = {"direct", "crop_preserve_aspect", None} # 'direct' means resize to the target size without preserving the aspect ratio

VALID_NORMALIZATION = {"gaussian", "minmax", None}

VALID_VOLUME_NORMALIZATION = {"minmax", None}

VALID_OUTPUT_FORMATS = {"npy", "png", "png16"}

def _deep_update(base, update):
    """
    Recursively update a nested dictionary.

    Parameters
    ----------
    base : dict
        Dictionary that will be updated in place.
    update : dict
        Values that overwrite those in ``base``.

    Returns
    -------
    dict
        The updated ``base`` dictionary.
    """
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base

def load_config(experiment_path):
    """
    Load an experiment config and merge it with its dataset config.

    The experiment file may contain a ``dataset`` key pointing to a dataset
    config (relative paths are resolved from the experiment file). Values are
    merged in this order, later ones overriding earlier ones:
    :data:`DEFAULTS` → dataset config → experiment config.

    It allows to have a single dataset config shared by multiple experiments, 
    and also to have a single experiment config that can be used with multiple datasets.

    Parameters
    ----------
    experiment_path : str or pathlib.Path
        Path to the experiment YAML file.

    Returns
    -------
    dict
        The merged and validated configuration.

    Raises
    ------
    FileNotFoundError
        If the experiment or dataset file does not exist.
    ValueError
        If the configuration contains invalid values.
    """
    experiment_path = Path(experiment_path)
    with open(experiment_path, "r", encoding="utf-8") as f:
        experiment = yaml.safe_load(f) or {}  # or {}: an empty file gives None

    config = deepcopy(DEFAULTS) # Start with the default values and don't modify the original dict.

    dataset_ref = experiment.pop("dataset", None) # The absolute path of the dataset is kept in _dataset_file.

    if dataset_ref is not None:
        dataset_path = (experiment_path.parent / dataset_ref).resolve()  # relative paths are converted to absolute paths
        with open(dataset_path, "r", encoding="utf-8") as f:
            _deep_update(config, yaml.safe_load(f) or {})
        config["_dataset_file"] = str(dataset_path) # Record which dataset file was used, as text

    _deep_update(config, experiment)
    config["_experiment_file"] = str(experiment_path.resolve()) # Record the absolute path of the experiment file
    config.setdefault("name", experiment_path.stem) # If no layer defined name, the experiment is named after its file

    validate_config(config)
    return config

def validate_config(config):
    """
    Check that a merged configuration is consistent.

    Only checks that do not require access to the data are performed here;
    paths are checked when each pipeline step starts.

    Parameters
    ----------
    config : dict
        Merged configuration, as returned by :func:`load_config`.

    Raises
    ------
    ValueError
        If a required key is missing or a value is not allowed.
    """
    errors = []

    if not config.get("bids_root"):
        errors.append("'bids_root' is required in the dataset config.")

    sl = config["slices"]
    if sl["view"] not in VALID_VIEWS:
        errors.append(f"slices.view must be one of {sorted(VALID_VIEWS)}.")
    if not isinstance(sl["n_slices"], int) or sl["n_slices"] < 1:
        errors.append("slices.n_slices must be a positive integer.")
    if sl["volume_normalization"] not in VALID_VOLUME_NORMALIZATION:
        errors.append("slices.volume_normalization must be 'minmax' or null.")
    if sl["output_format"] not in VALID_OUTPUT_FORMATS:
        errors.append(f"slices.output_format must be one of {sorted(VALID_OUTPUT_FORMATS)}.")

    resize = sl.get("resize") or {}
    if resize.get("method") not in VALID_RESIZE:
        errors.append("slices.resize.method must be 'direct', 'crop_preserve_aspect' or null.")

    norm = sl.get("normalization") or {}
    if norm.get("method") not in VALID_NORMALIZATION:
        errors.append("slices.normalization.method must be 'gaussian', 'minmax' or null.")

    if sl["source"] not in {"T1w", "tissue_maps"}:
        errors.append("slices.source must be 'T1w' or 'tissue_maps'.")

    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))

def save_config(config, path):
    """
    Save a configuration to a YAML file (e.g. next to the outputs).

    Parameters
    ----------
    config : dict
        Configuration to save.
    path : str or pathlib.Path
        Output YAML path. Parent folders are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Reading and writing images
# ---------------------------------------------------------------------------

#: File extension used for each output format.
EXTENSIONS = {"npy": ".npy", "png": ".png", "png16": ".png"}

def load_nifti(path, dtype=np.float32, canonical=True):
    """
    Load a NIfTI volume as a NumPy array.

    By default the volume is reoriented to RAS+ (closest canonical orientation),
    as in the original pipeline (``nib.as_closest_canonical``): the first array
    axis goes from left to right, the second from posterior to anterior and the
    third from inferior to superior. This only reorders and flips axes; voxel
    values are not interpolated. NaN and infinite values are replaced by 0.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.nii`` or ``.nii.gz`` file.
    dtype : numpy dtype, optional
        Output data type. Default is ``numpy.float32``.
    canonical : bool, optional
        Reorient to RAS+ before returning the data. Default is True.

    Returns
    -------
    data : numpy.ndarray
        Voxel intensities (in RAS+ orientation if ``canonical`` is True).
    zooms : tuple of float
        Voxel size in mm along each axis.
    """
    import nibabel as nib

    img = nib.load(str(path))
    if canonical:
        img = nib.as_closest_canonical(img)
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=dtype)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, tuple(float(z) for z in img.header.get_zooms()[:3])

def to_uint8(image):
    """
    Convert an image in [0, 1] to 8-bit integers in [0, 255].

    Parameters
    ----------
    image : numpy.ndarray
        Float image; values outside [0, 1] are clipped.

    Returns
    -------
    numpy.ndarray
        ``uint8`` image, rounded as in the original pipeline.
    """
    return (np.clip(image, 0.0, 1.0) * 255).round().astype(np.uint8)

def save_image(image, path, fmt="npy"):
    """
    Save a 2D (H, W) or multi-channel (H, W, C) image (values expected in [0, 1]),
    png/png16 clip values outside [0, 1] before converting them to integers;
    npy saves the values unchanged.

    Parameters
    ----------
    image : numpy.ndarray
        Image with values in [0, 1].
    path : str or pathlib.Path
        Output path. The extension is replaced to match ``fmt``.
    fmt : {"npy", "png", "png16"}, optional
        - ``"npy"``: float32 array, no loss of precision (recommended).
        - ``"png"``: 8-bit grayscale PNG, as in the original pipeline.
        - ``"png16"``: 16-bit grayscale PNG (single-channel images only).

    Returns
    -------
    pathlib.Path
        The path actually written.

    Raises
    ------
    ValueError
        If ``fmt`` is unknown, or ``png16`` is used with a multi-channel image.
    """
    if fmt not in EXTENSIONS:
        raise ValueError(f"Unknown format '{fmt}'. Use one of {sorted(EXTENSIONS)}.")

    path = Path(path)
    for suffix in [".nii.gz", ".npy", ".png"]:
        if path.name.endswith(suffix):
            path = path.with_name(path.name[: -len(suffix)])
            break
    path = path.with_name(path.name + EXTENSIONS[fmt])
    path.parent.mkdir(parents=True, exist_ok=True)

    image = np.asarray(image, dtype=np.float32)
    if fmt == "npy":
        np.save(path, image)
    elif fmt == "png":
        Image.fromarray(to_uint8(image)).save(path)
    else:
        if image.ndim != 2:
            raise ValueError("png16 only supports single-channel images.")
        arr = (np.clip(image, 0.0, 1.0) * 65535).round().astype(np.uint16)
        Image.fromarray(arr).save(path)
    return path

def load_image(path):
    """
    Load an image saved with :func:`save_image` as float32 in [0, 1].

    Parameters
    ----------
    path : str or pathlib.Path
        ``.npy`` or ``.png`` file.

    Returns
    -------
    numpy.ndarray
        Float32 image.
    """
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    with Image.open(path) as img:
        arr = np.array(img)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Running a step for many images (used by run_anat and run_slices)
# ---------------------------------------------------------------------------

def run_parallel(func, jobs, n_jobs):
    """
    Run ``func(**job)`` for each job, sequentially or in parallel.

    Returns a list of result dicts; exceptions are caught and reported as
    rows with ``status="error"``.
    """
    results = []
    if n_jobs == 1:
        for i, job in enumerate(jobs, 1):
            results.append(_safe_call(func, job))
            print(f"  [{i}/{len(jobs)}] sub-{job['entities']['subject']}: {results[-1]['status']}")
        return results

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(_safe_call, func, job): job for job in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            print(f"  [{i}/{len(jobs)}] sub-{futures[fut]['entities']['subject']}: " f"{results[-1]['status']}")
    return results

def _safe_call(func, job):
    """
    Run ``func(**job)`` and turn any error into a log row with status "error",
    so that one failing image does not stop the others.

    Defined at module level so that ProcessPoolExecutor can send it to other processes.
    """
    try:
        return func(**job)
    except Exception as exc:
        return {**job.get("entities", {}), "status": "error", "message": repr(exc),
                "traceback": traceback.format_exc()}

def write_log(rows, path):
    """Write the per-image log CSV and print a short summary."""
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    n_err = int((df["status"] == "error").sum()) if "status" in df else 0
    print(f"Log saved to {path} ({len(df) - n_err} ok, {n_err} errors).")
    return df
