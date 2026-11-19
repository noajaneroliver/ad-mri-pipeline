"""
admri: preprocessing and deep learning tools for structural MRI in BIDS format.

The package is organised in four modules:

- :mod:`admri.common`  configuration, reading/writing images and running a step
                       for many images (shared by both steps)
- :mod:`admri.bids`    find input images and build derivative paths (pybids)
- :mod:`admri.anat`    step ``anat``: skull stripping, registration to a template
                       and warping of the tissue maps
- :mod:`admri.slices`  step ``slices``: 2D slices/patches, resize and normalisation
"""

__version__ = "0.1.0"
