from __future__ import annotations

import argparse

from ml.datasets.rdd2022 import build_matrices, matrices_exist
from ml.defect_classifier.model import save_classifier, train_defect_classifier
from ml.defect_classifier.severity_regressor import save_severity_regressor, train_severity_regressor
from ml.evaluation.run import run_evaluation
from ml.severity_forecaster.model import save_forecaster, train_severity_forecaster


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RDD2022 matrices and train ML models")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip matrix build (train from existing matrices only)",
    )
    parser.add_argument(
        "--rebuild-matrices",
        action="store_true",
        help="Force rebuild of training matrices from RDD2022",
    )
    args = parser.parse_args()

    if not args.skip_download:
        if args.rebuild_matrices or not matrices_exist():
            print("Building training matrices from RDD2022...")
            build_matrices()
        else:
            print("Using existing training matrices")

    print("\nTraining defect classifier...")
    classifier = train_defect_classifier()
    classifier_path = save_classifier(classifier)
    print(f"Defect classifier saved to {classifier_path}")

    print("\nTraining severity regressor (damage area -> severity)...")
    severity_regressor = train_severity_regressor()
    regressor_path = save_severity_regressor(severity_regressor)
    print(f"Severity regressor saved to {regressor_path}")

    print("\nTraining severity forecaster...")
    forecaster = train_severity_forecaster()
    forecaster_path = save_forecaster(forecaster)
    print(f"Severity forecaster saved to {forecaster_path}")

    print("\nGenerating evaluation reports (confusion matrix, plots, metrics)...")
    metrics_path = run_evaluation()
    print(f"Metrics saved to {metrics_path}")

    print("\nAll models trained on RDD2022 matrices.")


if __name__ == "__main__":
    main()
