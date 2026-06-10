from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from ml.config import MATRICES_DIR
from ml.datasets.rdd2022 import RDDAnnotation, _crop_annotation, download_country


def main() -> None:
    test_samples_path = MATRICES_DIR / "test_samples.json"
    if not test_samples_path.exists():
        raise FileNotFoundError(
            "Test split samples not found. Run: python -m ml.build_matrices --force"
        )

    test_samples = json.loads(test_samples_path.read_text(encoding="utf-8"))
    output_dir = Path(__file__).resolve().parent / "sample_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    by_type: dict[str, dict] = {}
    for sample in test_samples:
        defect_type = sample["defect_type"]
        if defect_type not in by_type:
            by_type[defect_type] = sample

    if not any(Path(sample["image_path"]).exists() for sample in test_samples[:1]):
        download_country("Czech")

    for defect_type, sample in by_type.items():
        image_path = Path(sample["image_path"])
        if not image_path.exists():
            print(f"Missing image for {defect_type}: {image_path}")
            continue

        annotation = RDDAnnotation(
            image_path=image_path,
            country="Czech",
            rdd_label="",
            defect_type=defect_type,
            xmin=int(sample["xmin"]),
            ymin=int(sample["ymin"]),
            xmax=int(sample["xmax"]),
            ymax=int(sample["ymax"]),
        )
        image = Image.open(image_path).convert("RGB")
        crop = _crop_annotation(image, annotation)
        filename = defect_type.lower().replace(" ", "_") + ".jpg"
        crop.save(output_dir / filename, format="JPEG", quality=90)
        print(f"Wrote hold-out test sample {output_dir / filename}")


if __name__ == "__main__":
    main()
