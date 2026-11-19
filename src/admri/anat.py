"""
Anatomical preprocessing of T1-weighted images, one function per step:

1. :func:`skull_strip_subject`: brain extraction (HD-BET).
2. :func:`register_brain_subject`: registration to the template (ANTs SyN).
3. :func:`warp_tissue_maps_subject`: warping of the FAST tissue maps to the
   template space with the transforms of step 2 (optional).

:func:`preprocess_anat_subject` runs the three steps for one image, and
:func:`run_anat` runs them for all the selected images (``--step anat``).

Steps 1 and 2 use the group's preprocessing functions (``fun_defs.py`` from the
BIG "Scripts_Preproc_Homebrew" folder); their location is given in the config
(``group_tools.path``).

All outputs are renamed with BIDS derivative names:

- ``..._desc-brain_T1w.nii.gz``                       brain in native space
- ``..._space-<tpl>_desc-brain_T1w.nii.gz``           brain in template space
- ``..._from-T1w_to-<tpl>_desc-warp_xfm.nii.gz``      non-linear warp
- ``..._from-T1w_to-<tpl>_desc-affine_xfm.mat``       affine transform
- ``..._space-<tpl>_label-<GM|WM|CSF>_probseg.nii.gz`` tissue maps (optional)
"""

import glob
import os
import shutil # python library to move files and directories, "shell utilities"
import sys
from contextlib import contextmanager
from pathlib import Path

from .bids import derivative_path, ensure_dataset_description, get_derivatives_root, list_images
from .common import run_parallel, save_config, write_log

#: Tissue class of each FSL FAST partial volume estimate.
PVE_LABELS = {0: "CSF", 1: "GM", 2: "WM"}

#: File name used by FSL BET-style tools for each skull stripping method.
SKULL_STRIP_METHODS = {"hd-bet", "bet-nipype", "bet-fslpy"}


@contextmanager
def _quiet():
    """Silence stdout/stderr of external tools (HD-BET is very verbose)."""
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def load_group_tools(path, bin_path=None):
    """
    Import the group's preprocessing module (``fun_defs``).

    Parameters
    ----------
    path : str or pathlib.Path
        Folder that contains ``fun_defs.py`` and ``logwrapper.py``
        (config: ``group_tools.path``).
    bin_path : str or pathlib.Path, optional
        Folder prepended to ``PATH`` so that command line tools such as
        ``hd-bet`` are found (e.g. the bin folder of the conda environment;
        config: ``group_tools.bin_path``).

    Returns
    -------
    module
        The imported ``fun_defs`` module.

    Raises
    ------
    FileNotFoundError
        If ``fun_defs.py`` is not in ``path``.
    """
    path = Path(path)
    if not (path / "fun_defs.py").is_file():
        raise FileNotFoundError(f"fun_defs.py not found in {path}.")
    if str(path) not in sys.path:
        sys.path.append(str(path))
    if bin_path:
        os.environ["PATH"] = str(bin_path) + os.pathsep + os.environ.get("PATH", "")

    import fun_defs  # noqa: E402  (module provided by the group)

    return fun_defs


def skull_strip(input_path, output_dir, method="hd-bet", tools=None, quiet=True):
    """
    Remove non-brain tissue from a T1-weighted image.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Native T1w NIfTI file.
    output_dir : str or pathlib.Path
        Folder where the group function writes its output.
    method : {"hd-bet", "bet-nipype", "bet-fslpy"}, optional
        Skull stripping method. Default is ``"hd-bet"``.
    tools :Group module returned by :func:`load_group_tools`.
    quiet : bool, optional
        Hide the console output of the tool. Default is True.

    Returns
    -------
    pathlib.Path
        Path to the skull-stripped image (name chosen by the group function).

    Raises
    ------
    ValueError
        If ``method`` is unknown or ``tools`` is None.
    """
    if method not in SKULL_STRIP_METHODS:
        raise ValueError(f"Unknown skull stripping method '{method}'.")

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = (str(input_path.parent) + "/", input_path.name, str(output_dir) + "/")

    if method == "hd-bet":
        if quiet:
            with _quiet():
                out = tools.skull_stripping_hd_bet(*args)
        else:
            out = tools.skull_stripping_hd_bet(*args)
    elif method == "bet-nipype":
        out = tools.skull_stripping_nipype_bet(*args)
    else: # "bet-fslpy"
        out = tools.skull_stripping_fsl_bet(*args, 0.7)

    out = Path(out)
    return out if out.is_absolute() else output_dir / out


def register_to_template(moving_path, output_path, template_path, tools):
    """
    Register a skull-stripped T1w image to a template with the group tools.

    Uses the group function ``ants_registration_subject_T1_2_template`` (ANTs
    SyN).

    Parameters
    ----------
    moving_path : str or pathlib.Path
        Skull-stripped T1w image in native space.
    output_path : str or pathlib.Path
        Registered image to write (``.nii.gz``).
    template_path : str or pathlib.Path
        Fixed template, e.g. ``MNI152_T1_1mm_brain.nii.gz``.
    tools : module
        Group module (``fun_defs``), as returned by :func:`load_group_tools`.

    Returns
    -------
    dict
        ``{"image": path, "warp": path, "affine": path}`` with the registered
        image and the forward transforms (native -> template).

    Raises
    ------
    FileNotFoundError
        If the expected transform files are not produced.
    """
    moving_path, output_path = Path(moving_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = str(output_path)[: -len(".nii.gz")]

    tools.ants_registration_subject_T1_2_template(
        name_template=str(template_path),
        name_mov_t1=str(moving_path),
        name_t1_reg=str(output_path),
    )
    warp, affine = prefix + "_transform0.nii.gz", prefix + "_transform1.mat"

    for path in (output_path, warp, affine):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Expected registration output not found: {path}")
    return {"image": output_path, "warp": Path(warp), "affine": Path(affine)}


def apply_transforms(moving_path, output_path, template_path, warp_path, affine_path,
                     interpolation="linear"):
    """
    Apply existing native -> template transforms to another image.

    Used for the tissue probability maps, which are in the same native space as
    the T1w image, so no new registration is computed.

    Parameters
    ----------
    moving_path : str or pathlib.Path
        Image in native T1w space (e.g. a FAST partial volume estimate).
    output_path : str or pathlib.Path
        Output image in template space.
    template_path : str or pathlib.Path
        Template that defines the output grid.
    warp_path, affine_path : str or pathlib.Path
        Forward transforms returned by :func:`register_to_template`.
    interpolation : str, optional
        ANTs interpolator. Default is ``"linear"`` (suitable for probability maps).

    Returns
    -------
    pathlib.Path
        The written output path.
    """
    import ants

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    warped = ants.apply_transforms(
        fixed=ants.image_read(str(template_path)),
        moving=ants.image_read(str(moving_path)),
        transformlist=[str(warp_path), str(affine_path)],
        whichtoinvert=[False, False],
        interpolator=interpolation,
    )
    ants.image_write(warped, str(output_path))
    return output_path


def anat_outputs(derivatives_root, entities, space="MNI152NLin6Asym"):
    """
    Return the BIDS paths of all anatomical preprocessing outputs of one image.

    Parameters
    ----------
    derivatives_root : str or pathlib.Path
        Root of the pipeline derivatives (output folder).
    entities : dict
        Source entities of the input image (subject, session, run...).
    space : str, optional
        Template space label. Default is ``"MNI152NLin6Asym"`` (FSL MNI152).

    Returns
    -------
    dict of pathlib.Path
        Keys ``brain_native``, ``brain_template``, ``warp``, ``affine`` and
        ``probseg_<label>`` for each tissue class.
    """
    def path(output_entities, suffix, ext):
        return derivative_path(derivatives_root, {**entities, **output_entities}, suffix, ext)

    xfm = {"from": "T1w", "to": space} # 'xfm' = 'transform' abbreviation
    out = {
        "brain_native": path({"description": "brain"}, "T1w", ".nii.gz"),
        "brain_template": path({"space": space, "description": "brain"}, "T1w", ".nii.gz"),
        "warp": path({**xfm, "description": "warp"}, "xfm", ".nii.gz"),
        "affine": path({**xfm, "description": "affine"}, "xfm", ".mat"),
    }
    for label in PVE_LABELS.values(): # {0: "CSF", 1: "GM", 2: "WM"}
        out[f"probseg_{label}"] = path({"space": space, "label": label}, "probseg", ".nii.gz")
    return out


def find_tissue_map_files(segmentation_dir, pattern, entities):
    """
    Find the native FAST tissue maps of one image.

    Looks for the three FAST classes defined in :data:`PVE_LABELS` (0 = CSF, 1 = GM, 2 = WM).

    Parameters
    ----------
    segmentation_dir : str
        Folder with the segmentations. May contain ``{subject}`` and
        ``{session}`` placeholders.
    pattern : str
        Glob pattern of one map, with ``{subject}``, ``{session}`` and ``{pve}``
        placeholders, e.g. ``"sub-{subject}_ses-{session}*_T1w_fast_seg_pve_{pve}*.nii.gz"``.
    entities : dict
        Source entities of the T1w image.

    Returns
    -------
    dict
        ``{label_name: path}``, e.g. ``{"CSF": ..., "GM": ..., "WM": ...}``.

    Raises
    ------
    FileNotFoundError
        If a map is missing or several files match.
    """
    subj_values = {"subject": entities["subject"], "session": entities.get("session", "")}
    folder = segmentation_dir.format(**subj_values)
    out = {}
    for pve, label in PVE_LABELS.items(): # (0, "CSF"), (1, "GM"), (2, "WM")
        matches = sorted(glob.glob(os.path.join(folder, pattern.format(pve=pve, **subj_values))))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected 1 file for pve_{pve} in {folder}, found {len(matches)}."
            )
        out[label] = Path(matches[0])
    return out


def skull_strip_subject(t1_file, outputs, config, tools, overwrite=False):
    """
    Step 1: brain extraction of one T1w image.

    Parameters
    ----------
    t1_file : str or pathlib.Path
        Native T1w image selected from the BIDS dataset.
    outputs : dict of pathlib.Path
        Output paths, as returned by :func:`anat_outputs`.
    config : dict
        Merged configuration; uses ``skull_strip.method``.
    tools : module
        Group module (``fun_defs``), as returned by :func:`load_group_tools`.
    overwrite : bool, optional
        Recompute the output if it already exists. Default is False.

    Returns
    -------
    pathlib.Path
        ``outputs["brain_native"]``: the skull-stripped brain in native space.
    """
    out = outputs["brain_native"]
    if out.is_file() and not overwrite:
        return out

    # The group function chooses the output name, so it writes into a temporary
    # folder and the result is then moved to its BIDS name.
    work = out.parent / "_work_skullstrip"
    brain = skull_strip(t1_file, work, config["skull_strip"]["method"], tools)
    shutil.move(str(brain), out)
    shutil.rmtree(work, ignore_errors=True)
    return out


def register_brain_subject(outputs, config, tools, overwrite=False):
    """
    Step 2: register the skull-stripped brain to the template (ANTs SyN).

    Reads ``outputs["brain_native"]``, so :func:`skull_strip_subject` must
    have been run before.

    Parameters
    ----------
    outputs : dict of pathlib.Path
        Output paths, as returned by :func:`anat_outputs`.
    config : dict
        Merged configuration; uses ``registration.template``.
    tools : module
        Group module (``fun_defs``), as returned by :func:`load_group_tools`.
    overwrite : bool, optional
        Recompute the outputs if they already exist. Default is False.

    Returns
    -------
    dict of pathlib.Path
        ``outputs``; ``brain_template``, ``warp`` and ``affine`` are written.

    Raises
    ------
    FileNotFoundError
        If the skull-stripped brain does not exist.
    """
    keys = ("brain_template", "warp", "affine")
    if all(outputs[k].is_file() for k in keys) and not overwrite:
        return outputs
    if not outputs["brain_native"].is_file():
        raise FileNotFoundError(f"Skull-stripped brain not found: {outputs['brain_native']}")

    # The group function names the transforms after the registered image
    # (..._transform0.nii.gz, ..._transform1.mat), so it also works in a
    # temporary folder and the results are moved to their BIDS names.
    work = outputs["brain_template"].parent / "_work_registration"
    reg = register_to_template(outputs["brain_native"], work / "registered.nii.gz",
                               config["registration"]["template"], tools)
    for key in keys:
        src = reg["image"] if key == "brain_template" else reg[key]
        shutil.move(str(src), outputs[key])
    shutil.rmtree(work, ignore_errors=True)
    return outputs


def warp_tissue_maps_subject(entities, outputs, config, overwrite=False):
    """
    Step 3: bring the native FAST tissue maps (CSF, GM, WM) to template space.

    The maps are not computed here (FAST computed them before); they are warped
    with the T1w warp and affine of :func:`register_brain_subject`, so no new
    registration is computed.

    Parameters
    ----------
    entities : dict
        Source entities of the T1w image (used to find its FAST maps).
    outputs : dict of pathlib.Path
        Output paths, as returned by :func:`anat_outputs`.
    config : dict
        Merged configuration; uses ``tissue_maps`` and ``registration.template``.
    overwrite : bool, optional
        Recompute maps that already exist. Default is False.

    Returns
    -------
    dict of pathlib.Path
        ``outputs``; ``probseg_CSF``, ``probseg_GM`` and ``probseg_WM`` are written.
    """
    tm_cfg = config["tissue_maps"]
    # Find the native FAST maps of the subject
    maps = find_tissue_map_files(tm_cfg["segmentation_dir"], tm_cfg["pattern"], entities)

    # For each tissue class: if --overwrite is used or the registered map doesn't
    # exist yet, apply the T1 warp + affine to the native FAST map to bring it
    # to template space, and save it as the registered map.
    for label, path in maps.items():
        out = outputs[f"probseg_{label}"]
        if overwrite or not out.is_file():
            apply_transforms(path, out, config["registration"]["template"], outputs["warp"],
                             outputs["affine"], tm_cfg.get("interpolation", "linear"))
    return outputs


def preprocess_anat_subject(t1_file, entities, derivatives_root, config, tools=None,
                            overwrite=False):
    """
    Run all the anatomical preprocessing steps for one image.

    1. :func:`skull_strip_subject`
    2. :func:`register_brain_subject`
    3. :func:`warp_tissue_maps_subject` (only if ``tissue_maps.enabled``)

    Each step is skipped if its outputs already exist (unless ``overwrite``),
    so after a failure only the missing steps are run again.

    Parameters
    ----------
    t1_file : str or pathlib.Path
        Native T1w image selected from the BIDS dataset.
    entities : dict
        Source entities of the image (see :func:`admri.bids.source_entities`).
        Entities are the key-value pairs that make up a BIDS file name.
        Each key-value chunk separated by _ is one entity.
    derivatives_root : str or pathlib.Path
        Root of the pipeline derivatives.
    config : dict
        Merged configuration; uses the ``group_tools``, ``skull_strip``,
        ``registration``, ``tissue_maps`` and ``derivatives`` keys.
    tools : module, optional
        Group module; loaded from the config if not given.
    overwrite : bool, optional
        Recompute outputs that already exist. Default is False.

    Returns
    -------
    dict of pathlib.Path
        The outputs, as in :func:`anat_outputs`.
    """
    outputs = anat_outputs(derivatives_root, entities, config["derivatives"]["space"])
    if tools is None:
        tools = load_group_tools(config["group_tools"]["path"], config["group_tools"]["bin_path"])

    skull_strip_subject(t1_file, outputs, config, tools, overwrite)
    register_brain_subject(outputs, config, tools, overwrite)
    if config["tissue_maps"].get("enabled"):
        warp_tissue_maps_subject(entities, outputs, config, overwrite)
    return outputs


# ---------------------------------------------------------------------------
# Run the step for all images (--step anat)
# ---------------------------------------------------------------------------

def _anat_one(path, entities, derivatives_root, config, overwrite):
    """Preprocess one image (worker function for :func:`run_anat`)."""
    outputs = preprocess_anat_subject(path, entities, derivatives_root, config,
                                      overwrite=overwrite)
    return {**entities, "status": "ok", "message": "",
            "brain_template": str(outputs["brain_template"])}

def run_anat(config, start=None, end=None, n_jobs=None, overwrite=None):
    """
    Anatomical preprocessing of all selected images.

    Skull stripping, registration to the template and (if
    ``tissue_maps.enabled``) warping of the tissue maps; see
    :func:`admri.anat.preprocess_anat_subject`.

    Parameters
    ----------
    config : dict
        Merged configuration.
    start, end : int, optional
        Process only ``images[start:end]``, to split the work between several
        jobs launched at the same time.
    n_jobs : int, optional
        Number of parallel processes. Default is ``config["run"]["n_jobs"]``.
    overwrite : bool, optional
        Recompute existing outputs. Default is ``config["run"]["overwrite"]``.

    The configuration used is saved next to the log
    (``logs/anat_config_used_<start>-<end>.yaml``), with the ``n_jobs`` and
    ``overwrite`` values actually used (command line or config).

    Returns
    -------
    pandas.DataFrame
        Log with one row per image.
    """
    n_jobs = n_jobs or config["run"]["n_jobs"]
    overwrite = config["run"]["overwrite"] if overwrite is None else overwrite
    deriv = get_derivatives_root(config)
    ensure_dataset_description(deriv, config["derivatives"]["pipeline_name"])
    suffix = f"_{start or 0}-{end if end is not None else 'end'}"
    save_config({**config, "run": {"n_jobs": n_jobs, "overwrite": overwrite}},
                deriv / "logs" / f"anat_config_used{suffix}.yaml")

    images = list_images(config)[start:end]
    print(f"Anatomical preprocessing of {len(images)} images -> {deriv}")
    jobs = [{"path": im["path"], "entities": im["entities"], "derivatives_root": deriv,
             "config": config, "overwrite": overwrite} for im in images]
    rows = run_parallel(_anat_one, jobs, n_jobs)
    return write_log(rows, deriv / "logs" / f"anat{suffix}.csv")
