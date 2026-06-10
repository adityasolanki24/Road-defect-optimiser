from __future__ import annotations

import argparse

from ml.config import DEFAULT_RDD_COUNTRIES
from ml.datasets.rdd2022 import build_matrices, matrices_exist


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RDD2022 and build training matrices")
    parser.add_argument(
        "--countries",
        nargs="+",
        default=list(DEFAULT_RDD_COUNTRIES),
        help="RDD2022 country folders to include (default: Czech Japan)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on annotation samples",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild matrices even if they already exist",
    )
    args = parser.parse_args()

    if matrices_exist() and not args.force:
        print("Matrices already exist. Use --force to rebuild.")
        return

    stats = build_matrices(tuple(args.countries), max_samples=args.max_samples)
    print("Matrix build complete:", stats)


if __name__ == "__main__":
    main()
