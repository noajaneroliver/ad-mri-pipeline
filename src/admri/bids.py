"""
Find input images in a BIDS dataset and build BIDS-style derivative paths.

Input images are queried with pybids (:class:`bids.BIDSLayout`), so the same
code works for any dataset in BIDS format. Everything that is dataset-specific
(which session, which run to prefer, extra entity filters) is defined in the
config file.
"""

import json
from pathlib import Path

import pandas as pd

#: BIDS entities used in the file names, in the order required by BIDS, with the
#: short key written in the name. Other entities (e.g. ``view``, ``slice``,
#: ``patch`` of the 2D outputs) are added after these, in alphabetical order.
ENTITIES = {
    "subject": "sub", "session": "ses", "acquisition": "acq", "run": "run",
    "from": "from", "to": "to", "space": "space", "label": "label",
    "description": "desc",
}


def strip_prefix(label, prefix):
    """
    Remove a BIDS prefix from a label if present.

    Parameters
    ----------
    label : str
        Label such as ``"sub-ADNI002S0295"`` or ``"ADNI002S0295"``.
    prefix : str
        Prefix without dash, e.g. ``"sub"`` or ``"ses"``.

    Returns
    -------
    str
        Label without the prefix (``"ADNI002S0295"``).
    """
    label = str(label).strip()
    return label[len(prefix) + 1:] if label.startswith(prefix + "-") else label


def load_layout(bids_root, database_path=None):
    """
    Index a BIDS dataset with pybids.

    File names are not validated against the BIDS specification, because
    converted datasets often contain small irregularities that would make
    pybids ignore files.

    Parameters
    ----------
    bids_root : str or pathlib.Path
        Root folder of the BIDS dataset.
    database_path : str or pathlib.Path, optional
        Where to store the pybids index (SQLite). Indexing a large dataset takes
        minutes the first time; with a database the next runs load it in seconds.
        Delete this folder after adding or removing files in the dataset.

    Returns
    -------
    bids.BIDSLayout
        The indexed dataset.
    """
    from bids import BIDSLayout

    if database_path is None:
        return BIDSLayout(str(bids_root), validate=False)
    Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    return BIDSLayout(str(bids_root), validate=False, database_path=str(database_path))


def get_subjects(layout, subjects=None, subject_list=None, subject_column="subject"):
    """
    Return the list of subjects to process.

    Parameters
    ----------
    layout : bids.BIDSLayout
        Indexed BIDS dataset.
    subjects : list of str, optional
        Explicit list of subject labels (with or without ``sub-``).
    subject_list : str or pathlib.Path, optional
        CSV file with one subject per row. Used only if ``subjects`` is None.
    subject_column : str, optional
        Column of ``subject_list`` containing the subject labels.
        Default is ``"subject"``.

    Returns
    -------
    list of str
        Sorted subject labels without the ``sub-`` prefix. If neither
        ``subjects`` nor ``subject_list`` is given, all subjects in the dataset.

    Raises
    ------
    KeyError
        If ``subject_column`` is not a column of ``subject_list``.
    """
    available = set(layout.get_subjects())

    if subjects is None and subject_list is not None:
        df = pd.read_csv(subject_list)
        if subject_column not in df.columns:
            raise KeyError(f"Column '{subject_column}' not found in {subject_list}.")
        subjects = df[subject_column].dropna().astype(str).tolist()

    if subjects is None:
        return sorted(available)

    requested = [strip_prefix(s, "sub") for s in subjects]
    missing = sorted(set(requested) - available)
    if missing:
        print(f"[bids] {len(missing)} requested subjects are not in the dataset "
              f"(first: {missing[:5]}).")
    return sorted(s for s in dict.fromkeys(requested) if s in available)


def get_sessions(layout, subject, session=None):
    """
    Return the sessions to process for one subject.

    Parameters
    ----------
    layout : bids.BIDSLayout
        Indexed BIDS dataset.
    subject : str
        Subject label without ``sub-``.
    session : str, optional
        - ``None``: the first available session (alphabetical order), which
          gives one image per subject.
        - ``"all"``: every session of the subject.
        - any other value: that session (with or without ``ses-``).

    Returns
    -------
    list of str or list of None
        Session labels without ``ses-``. ``[None]`` if the dataset has no sessions.
    """
    available = sorted(layout.get_sessions(subject=subject))
    if not available:
        return [None]
    if session is None:
        return available[:1]
    if session == "all":
        return available
    session = strip_prefix(session, "ses")
    return [session] if session in available else []



def select_image(layout, subject, session=None, datatype="anat", suffix="T1w",
                 extension=".nii.gz", filters=None, prefer=None):
    """
    Select one image for a subject and session.

    Parameters
    ----------
    layout : bids.BIDSLayout
        Indexed BIDS dataset.
    subject : str
        Subject label without ``sub-``.
    session : str or None, optional
        Session label without ``ses-``. None if the dataset has no sessions.
    datatype : str, optional
        BIDS datatype folder. Default is ``"anat"``.
    suffix : str, optional
        BIDS suffix. Default is ``"T1w"``.
    extension : str, optional
        File extension. Default is ``".nii.gz"``.
    filters : dict, optional
        Extra pybids filters with pybids entity names, e.g.
        ``{"acquisition": "MPRAGE"}``.
    prefer : dict, optional
        Used when several files match: ``entity -> preferred values in order``.
        For example ``{"run": [2]}`` puts ``run-02`` first, then the rest
        (original ADNI rule). Files with other values go last; ties are broken
        alphabetically by file name. ``2``, ``"2"`` and ``"02"`` are equal.

    Returns
    -------
    bids.layout.BIDSFile or None
        The selected file, or None if no file matches.
    """
    query = {"subject": subject, "datatype": datatype, "suffix": suffix, "extension": extension,
             **(filters or {})}
    if session is not None:
        query["session"] = session
    files = layout.get(**query)
    if not files:
        return None

    def norm(value):  # 2, "2" and "02" -> "2"
        text = str(value)
        return str(int(text)) if text.isdigit() else text

    def rank(f):  # position of each file entity in its preference list
        entities = f.get_entities()
        ranks = []
        for entity, values in (prefer or {}).items():
            wanted = [norm(v) for v in (values if isinstance(values, list) else [values])]
            current = norm(entities.get(entity))
            ranks.append(wanted.index(current) if current in wanted else len(wanted))
        return (*ranks, Path(f.path).name)

    return sorted(files, key=rank)[0]


def build_bids_name(entities, suffix, extension):
    """
    Build a BIDS file name from a dictionary of entities.

    Parameters
    ----------
    entities : dict
        Entities with pybids (long) names, e.g.
        ``{"subject": "01", "session": "01", "space": "MNI152NLin6Asym"}``.
        Entities with value None are skipped.
    suffix : str
        BIDS suffix (always the last element), e.g. ``"T1w"``.
    extension : str
        Extension including the dot, e.g. ``".nii.gz"``.

    Returns
    -------
    str
        The file name, e.g. ``"sub-01_ses-01_space-MNI152NLin6Asym_T1w.nii.gz"``.

    Examples
    --------
    >>> build_bids_name({"subject": "01", "description": "brain"}, "T1w", ".nii.gz")
    'sub-01_desc-brain_T1w.nii.gz'
    """
    entities = {k: v for k, v in entities.items() if v is not None}
    parts = []
    for entity, key in ENTITIES.items():
        if entity in entities:
            value = entities[entity]
            if entity == "run" and str(value).isdigit():
                value = f"{int(value):02d}"
            parts.append(f"{key}-{value}")
    for entity in sorted(k for k in entities if k not in ENTITIES):
        parts.append(f"{entity}-{entities[entity]}")
    return "_".join(parts + [suffix]) + extension


def derivative_path(derivatives_root, entities, suffix, extension):
    """
    Build the full path of a derivative file in BIDS layout.

    Parameters
    ----------
    derivatives_root : str or pathlib.Path
        Root of the pipeline derivatives, e.g. ``<bids_root>/derivatives/admri``.
    entities : dict
        Entities of the file; must include ``subject``. See :func:`build_bids_name`.
    suffix : str
        BIDS suffix, e.g. ``"T1w"`` or ``"xfm"``.
    extension : str
        Extension including the dot.

    Returns
    -------
    pathlib.Path
        ``<derivatives_root>/sub-<label>[/ses-<label>]/anat/<file name>``.
    """
    folder = Path(derivatives_root) / f"sub-{entities['subject']}"
    if entities.get("session") is not None:
        folder = folder / f"ses-{entities['session']}"
    return folder / "anat" / build_bids_name(entities, suffix, extension)


def source_entities(bids_file):
    """
    Return the entities of an input file that identify it.

    Entities are the key–value pairs that make up a BIDS file name. 
    Each key-value chunk separated by _ is one entity

    These are copied to every derivative so that each output can be traced back
    to its source image.

    Parameters
    ----------
    bids_file : bids.layout.BIDSFile
        Input file returned by :func:`select_image`.

    Returns
    -------
    dict
        Long entity names (subject, session, acquisition, run...) and values.
    """
    keep = ["subject", "session", "acquisition", "run"]
    ents = bids_file.get_entities()
    return {k: ents[k] for k in keep if ents.get(k) is not None}



def ensure_dataset_description(derivatives_root, pipeline_name="admri"):
    """
    Create ``dataset_description.json`` in a derivatives folder if missing.

    pybids only indexes a derivatives folder when this file exists.

    Parameters
    ----------
    derivatives_root : str or pathlib.Path
        Root of the pipeline derivatives.
    pipeline_name : str, optional
        Name written in ``GeneratedBy``. Default is ``"admri"``.
    """
    root = Path(derivatives_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "dataset_description.json"
    if not path.exists():
        description = {
            "Name": f"{pipeline_name} derivatives",
            "BIDSVersion": "1.9.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": pipeline_name}],
        }
        path.write_text(json.dumps(description, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Selecting the images to process (used by run_anat and run_slices)
# ---------------------------------------------------------------------------

def get_derivatives_root(config):
    """
    Return the root folder of the pipeline derivatives.

    Parameters
    ----------
    config : dict
        Merged configuration.

    Returns
    -------
    pathlib.Path
        ``config["derivatives_root"]`` if set, otherwise
        ``<bids_root>/derivatives/<pipeline_name>``.
    """
    if config.get("derivatives_root"):
        return Path(config["derivatives_root"])
    return Path(config["bids_root"]) / "derivatives" / config["derivatives"]["pipeline_name"]

def list_images(config):
    """
    List the input T1w images to process, one per subject and session.

    Parameters
    ----------
    config : dict
        Merged configuration (``bids_root`` and ``bids`` keys).

    Returns
    -------
    list of dict
        One dict per image with keys ``path`` (str) and ``entities`` (dict).
        Subjects or sessions without a matching image are reported and skipped.
    """
    b = config["bids"]
    layout = load_layout(config["bids_root"], b["database_path"])

    images = []
    subjects = get_subjects(layout, b["subjects"], b["subject_list"], b["subject_column"])
    for subject in subjects:
        for session in get_sessions(layout, subject, b["session"]):
            f = select_image(
                layout, subject, session, b["datatype"], b["suffix"], b["extension"],
                b["filters"], b["prefer"],
            )
            if f is None:
                print(f"[list_images] No {b['suffix']} for sub-{subject} ses-{session}.")
                continue
            images.append({"path": f.path, "entities": source_entities(f)})
    return images
