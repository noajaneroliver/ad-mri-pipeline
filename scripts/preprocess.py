"""
Command line entry point for the preprocessing pipeline.

Examples
--------
Anatomical preprocessing of all subjects (skull stripping, registration and
tissue maps; once per dataset)::

    python scripts/preprocess.py --config configs/experiments/anat.yaml --step anat

Split it in blocks launched at the same time::

    python scripts/preprocess.py --config ... --step anat --start 0 --end 200
    python scripts/preprocess.py --config ... --step anat --start 200 --end 400

Extract the 2D inputs of one experiment (repeat for each input configuration)::

    python scripts/preprocess.py --config configs/experiments/coronal_3slices.yaml --step slices --n-jobs 8
"""

import argparse

from admri.anat import run_anat
from admri.common import load_config
from admri.slices import run_slices


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MRI preprocessing pipeline (BIDS).")
    parser.add_argument("--config", required=True, help="Experiment YAML file.")
    parser.add_argument("--step", required=True, choices=["anat", "slices", "all"],
                        help="Pipeline step to run.")
    parser.add_argument("--start", type=int, default=None, help="First image index (inclusive).")
    parser.add_argument("--end", type=int, default=None, help="Last image index (exclusive).")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel processes.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing outputs.")
    return parser.parse_args()


def main():
    """Load the config and run the requested step(s)."""
    args = parse_args()
    config = load_config(args.config)

    if args.step in ("anat", "all"):
        run_anat(config, args.start, args.end, args.n_jobs, overwrite=True if args.overwrite else None)
    if args.step in ("slices", "all"):
        run_slices(config, args.start, args.end, args.n_jobs)


if __name__ == "__main__":
    main()
