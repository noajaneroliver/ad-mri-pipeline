"""
Step ``slices``: build the 2D inputs of the model from the 3D outputs of ``anat``.

For each image (:func:`run_slices` → :func:`_slices_one`):

1. load the registered volume (or the 3 tissue maps), reoriented to RAS+;
2. 3D min-max normalisation of the brain;
3. extract the central slices (and, optionally, the patches);
4. resize and normalise each 2D image;
5. save it, with one row per image in ``images.csv``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

from .anat import PVE_LABELS, anat_outputs
from .bids import derivative_path, get_derivatives_root, list_images
from .common import load_nifti, run_parallel, save_config, save_image, to_uint8, write_log


# ---------------------------------------------------------------------------
# Slices and patches from 3D volumes
# ---------------------------------------------------------------------------

#: Array axis of each anatomical view for volumes in RAS+ orientation
#: (as returned by :func:`admri.common.load_nifti`: x = sagittal, y = coronal, z = axial).
VIEW_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}

def normalize_volume(volume, threshold=0.0):
    """
    Min-max normalise a volume to [0, 1] using only voxels above a threshold.

    Voxels at or below the threshold (the background) are set to 0.

    Parameters
    ----------
    volume : numpy.ndarray
        3D volume.
    threshold : float, optional
        Voxels with intensity greater than this value are used to compute the
        minimum and maximum. Default is 0.0.

    Returns
    -------
    numpy.ndarray
        Float32 volume with values in [0, 1].
    """
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    mask = volume > threshold
    out = np.zeros_like(volume, dtype=np.float32)
    if np.any(mask):
        vals = volume[mask]
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            out[mask] = (vals - vmin) / (vmax - vmin)
    return out

def slice_indices(shape, view="coronal", n_slices=3, center=None):
    """
    Return the indices of the central slices along one view.

    Parameters
    ----------
    shape : tuple of int
        Shape of the 3D volume.
    view : {"axial", "coronal", "sagittal"}, optional
        Anatomical view. Default is ``"coronal"``.
    n_slices : int, optional
        Number of consecutive slices. Default is 3.
    center : int, optional
        Index of the central slice. Default is ``shape[axis] // 2``.

    Returns
    -------
    list of int
        ``n_slices`` consecutive indices around the centre. For an odd number
        the centre is in the middle (3 slices: ``c-1, c, c+1``); for an even
        number there is one more slice before the centre
        (20 slices: ``c-10 ... c+9``).

    Raises
    ------
    ValueError
        If the view is unknown or the indices fall outside the volume.

    Examples
    --------
    >>> slice_indices((182, 218, 182), "coronal", 3)
    [108, 109, 110]
    """
    if view not in VIEW_AXIS:
        raise ValueError(f"Unknown view '{view}'. Use one of {sorted(VIEW_AXIS)}.")
    size = shape[VIEW_AXIS[view]]
    if center is None:
        center = size // 2
    start = center - n_slices // 2
    indices = list(range(start, start + n_slices))
    if indices[0] < 0 or indices[-1] >= size:
        raise ValueError(f"Slices {indices[0]}..{indices[-1]} out of bounds for size {size}.")
    return indices

def extract_slice(volume, view, index, rotate90=True):
    """
    Extract one 2D slice from a 3D volume.

    Parameters
    ----------
    volume : numpy.ndarray
        3D volume.
    view : {"axial", "coronal", "sagittal"}
        Anatomical view.
    index : int
        Slice index along the axis of the view.
    rotate90 : bool, optional
        Rotate the slice 90 degrees counter-clockwise so that it is displayed
        in radiological orientation (superior part up). Default is True, as in
        the original pipeline.

    Returns
    -------
    numpy.ndarray
        2D float32 slice.
    """
    sl = np.take(volume, index, axis=VIEW_AXIS[view])
    if rotate90:
        sl = np.rot90(sl)
    return np.ascontiguousarray(np.nan_to_num(sl, nan=0.0), dtype=np.float32)

def extract_slices(volume, view="coronal", n_slices=3, center=None, rotate90=True):
    """
    Extract the central slices of a volume along one view.

    Parameters
    ----------
    volume : numpy.ndarray
        3D volume.
    view : {"axial", "coronal", "sagittal"}, optional
        Anatomical view. Default is ``"coronal"``.
    n_slices : int, optional
        Number of consecutive central slices. Default is 3.
    center : int, optional
        Index of the central slice. Default is the middle of the axis.
    rotate90 : bool, optional
        See :func:`extract_slice`. Default is True.

    Returns
    -------
    list of tuple
        ``(index, slice_2d)`` pairs in increasing index order.
    """
    indices = slice_indices(volume.shape, view, n_slices, center)
    return [(i, extract_slice(volume, view, i, rotate90)) for i in indices]

def extract_patches(image, patch_size=64, margin_bottom=28, margin_side=12, mirror_left=True):
    """
    Extract the two inferior lateral patches of a coronal slice.

    These patches cover the temporal lobes. The left patch is mirrored so that
    both patches have the same orientation and can be used as equivalent samples.

    Parameters
    ----------
    image : numpy.ndarray
        2D slice (before resizing; 182 x 182 in MNI 1 mm space).
    patch_size : int, optional
        Side of the square patch. Default is 64.
    margin_bottom : int, optional
        Pixels between the patch and the lower border. Default is 28.
    margin_side : int, optional
        Pixels between the patch and the lateral border. Default is 12.
    mirror_left : bool, optional
        Flip the left patch horizontally. Default is True.

    Returns
    -------
    dict
        ``{"right": patch, "leftflip": patch}`` (or ``"left"`` when
        ``mirror_left`` is False), each of shape ``(patch_size, patch_size)``.
    """
    h, w = image.shape[:2]
    y1 = h - margin_bottom                     # rows: same for both patches
    y0 = y1 - patch_size
    out = {"right": image[y0:y1, w - margin_side - patch_size:w - margin_side].copy()}
    left = image[y0:y1, margin_side:margin_side + patch_size]
    if mirror_left:
        out["leftflip"] = left[:, ::-1].copy()
    else:
        out["left"] = left.copy()
    return out


# ---------------------------------------------------------------------------
# Resize and intensity normalisation of 2D images
# ---------------------------------------------------------------------------

#: Ratio between the FWHM and the standard deviation of a Gaussian,
#: with the same rounding as in the original pipeline (exact value 2.3548...).
FWHM_TO_SIGMA = 2.355

def _pil_resize(image, size, legacy_uint8):
    """Resize one 2D float image with Lanczos interpolation (size = (width, height))."""
    if legacy_uint8:
        pil = Image.fromarray((np.clip(image, 0, 1) * 255).round().astype(np.uint8), mode="L")
        return np.asarray(pil.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    pil = Image.fromarray(np.asarray(image, dtype=np.float32), mode="F")
    return np.asarray(pil.resize(size, Image.Resampling.LANCZOS), dtype=np.float32)

def resize(image, size=224, method="crop_preserve_aspect", legacy_uint8=False):
    """
    Resize a 2D image to a square of ``size`` x ``size`` pixels.

    Parameters
    ----------
    image : numpy.ndarray
        2D float image.
    size : int, optional
        Output side in pixels. Default is 224 (ResNet input size).
    method : {"crop_preserve_aspect", "direct", None}, optional
        - ``"crop_preserve_aspect"``: scale so that the image covers the whole
          square while keeping its proportions, then crop the centre. Avoids
          distorting rectangular slices (e.g. axial 218 x 182).
        - ``"direct"``: resize without keeping proportions (fine for square images).
        - ``None``: return the image unchanged.
        Default is ``"crop_preserve_aspect"``. For square images both methods
        give the same result.
    legacy_uint8 : bool, optional
        Resize an 8-bit version of the image, exactly as the original pipeline
        did with PNG files. Default is False (resize in float, no precision loss).

    Returns
    -------
    numpy.ndarray
        Resized float32 image. Lanczos interpolation can produce values slightly
        outside [0, 1]; they are clipped.

    Raises
    ------
    ValueError
        If ``method`` is unknown.
    """
    if method is None:
        return np.asarray(image, dtype=np.float32)

    h, w = image.shape
    if method == "direct":
        out = _pil_resize(image, (size, size), legacy_uint8)
    elif method == "crop_preserve_aspect":
        scale = max(size / w, size / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        out = _pil_resize(image, (new_w, new_h), legacy_uint8)
        left, top = (new_w - size) // 2, (new_h - size) // 2
        out = out[top:top + size, left:left + size]
    else:
        raise ValueError(f"Unknown resize method '{method}'.")
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def gaussian_normalize(image, fwhm=6.0, pixel_size=1.0, background_threshold=0.0):
    """
    Normalise intensities using limits estimated on a smoothed image.

    The image is smoothed with a Gaussian filter only to obtain stable
    intensity limits inside the brain; the original (not smoothed) image is
    then rescaled:

    .. math:: I_{norm} = \\frac{I - v_{min}}{v_{max} - v_{min}}

    where :math:`v_{min}` and :math:`v_{max}` are the minimum and maximum of the
    smoothed image inside the tissue mask. The result is clipped to [0, 1].

    Parameters
    ----------
    image : numpy.ndarray
        2D float image in [0, 1].
    fwhm : float, optional
        Full width at half maximum of the Gaussian filter, in mm. Default is 6.0.
    pixel_size : float, optional
        Size of one pixel in mm, used to convert the FWHM into pixels. Default is
        1.0, as in the original pipeline (which applied 6 / 2.355 pixels even
        after resizing). For a 182 -> 224 resize the real size is 182/224 ≈ 0.81 mm.
    background_threshold : float, optional
        Pixels above this value form the tissue mask. Default is 0.0.

    Returns
    -------
    numpy.ndarray
        Normalised float32 image in [0, 1].

    Raises
    ------
    ValueError
        If the image has no pixels above the threshold or the limits are equal.
    """
    image = np.asarray(image, dtype=np.float32)
    mask = image > background_threshold
    if not np.any(mask):
        raise ValueError("No pixels above the background threshold.")

    sigma = fwhm / FWHM_TO_SIGMA / pixel_size
    smooth = gaussian_filter(image, sigma=sigma)
    vals = smooth[mask]
    vmin, vmax = vals.min(), vals.max()
    if vmax <= vmin:
        raise ValueError("vmax <= vmin after smoothing; cannot normalise.")
    return np.clip((image - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)

# ---------------------------------------------------------------------------
# Run the step for all images (--step slices)
# ---------------------------------------------------------------------------

def preprocess_slice(image, sl_cfg):
    """
    Resize and normalise one 2D image (slice or patch).

    Parameters
    ----------
    image : numpy.ndarray
        2D slice or patch.
    sl_cfg : dict
        The ``slices`` section of the config.

    Returns
    -------
    numpy.ndarray
        Processed float32 image in [0, 1].

    Raises
    ------
    ValueError
        If the normalisation method is unknown.
    """
    legacy = sl_cfg["legacy_uint8"]
    if legacy:
        # The original pipeline saved each extracted slice as an 8-bit PNG.
        image = to_uint8(image) / np.float32(255)

    resize_cfg = sl_cfg.get("resize") or {}
    out = resize(image, resize_cfg.get("size", 224), resize_cfg.get("method"), legacy)

    norm_cfg = sl_cfg.get("normalization") or {}
    method = norm_cfg.get("method")
    threshold = norm_cfg.get("background_threshold", sl_cfg["background_threshold"])
    if method == "gaussian":
        return gaussian_normalize(out, norm_cfg.get("fwhm", 6.0), norm_cfg.get("pixel_size", 1.0), threshold)
    if method == "minmax":
        return normalize_volume(out, threshold)
    if method is None:                         # e.g. tissue maps, already in [0, 1]
        return np.asarray(out, dtype=np.float32)
    raise ValueError(f"Unknown normalisation method '{method}'.")

def _slices_one(path, entities, derivatives_root, out_root, config):
    """Extract and save all images of one subject (worker for :func:`run_slices`)."""
    sl = config["slices"]
    space = config["derivatives"]["space"]
    outputs = anat_outputs(derivatives_root, entities, space)

    if sl["source"] == "T1w":
        sources = {None: outputs["brain_template"]}
    else:
        sources = {label: outputs[f"probseg_{label}"] for label in PVE_LABELS.values()}

    rows = []
    for label, src in sources.items():
        if not src.is_file():
            raise FileNotFoundError(f"Registered image not found: {src}")
        volume, _ = load_nifti(src)
        if sl["source"] == "T1w" and sl["volume_normalization"] == "minmax":
            volume = normalize_volume(volume, sl["background_threshold"])

        # central slice index, used to store each slice's position (-1, 0, +1...)
        center = sl["center"] if sl["center"] is not None else volume.shape[VIEW_AXIS[sl["view"]]] // 2
        for idx, image in extract_slices(volume, sl["view"], sl["n_slices"], sl["center"], sl["rotate90"]):
            patches = (extract_patches(image, **sl["patches"]) if sl.get("patches")
                       else {None: image})
            for patch_name, img in patches.items():
                out = preprocess_slice(img, sl)
                ents = {**entities, "space": space, "label": label,
                        "view": sl["view"], "slice": f"{idx:03d}", "patch": patch_name}
                suffix = "T1w" if sl["source"] == "T1w" else "probseg"
                path_out = derivative_path(out_root, ents, suffix, ".npy")
                path_out = save_image(out, path_out, sl["output_format"])
                rows.append({**entities, "source": sl["source"], "label": label,
                             "view": sl["view"], "slice_index": idx,
                             "slice_position": idx - center,
                             "patch": patch_name, "path": str(path_out)})
    return {**entities, "status": "ok", "message": "", "n_images": len(rows), "rows": rows}

def run_slices(config, start=None, end=None, n_jobs=None):
    """
    Extract, resize, normalise and save the 2D inputs of all subjects.

    The output folder receives a copy of the config, a ``images.csv`` with one
    row per saved image (``slice_position`` is -1/0/+1 for 3 slices, which tells
    the model which image goes to each channel) and a log CSV.

    Parameters
    ----------
    config : dict
        Merged configuration.
    start, end : int, optional
        Process only ``images[start:end]``.
    n_jobs : int, optional
        Number of parallel processes. Default is ``config["run"]["n_jobs"]``.

    Returns
    -------
    pandas.DataFrame
        Catalog of saved images.
    """
    n_jobs = n_jobs or config["run"]["n_jobs"]
    deriv = get_derivatives_root(config)
    # output folder: slices.output_dir if set, otherwise <derivatives_root>/slices/<experiment name>
    out_root = (Path(config["slices"]["output_dir"]) if config["slices"].get("output_dir")
                else deriv / "slices" / config["name"])
    save_config(config, out_root / "config_used.yaml")

    images = list_images(config)[start:end]
    print(f"Extracting slices for {len(images)} images -> {out_root}")
    jobs = [{"path": im["path"], "entities": im["entities"], "derivatives_root": deriv,
             "out_root": out_root, "config": config} for im in images]
    results = run_parallel(_slices_one, jobs, n_jobs)

    suffix = "" if start is None and end is None else f"_{start or 0}-{end if end is not None else 'end'}"
    images = pd.DataFrame([r for res in results for r in res.pop("rows", [])])
    images.to_csv(out_root / f"images{suffix}.csv", index=False)
    write_log(results, out_root / f"log{suffix}.csv")
    return images
