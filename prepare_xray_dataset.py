import csv
import hashlib
import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path("data/raw/xray/hf_data")
OUTPUT = Path("data/processed/xray")
IMAGES = OUTPUT / "images"
IMAGES.mkdir(parents=True, exist_ok=True)

manifest_path = OUTPUT / "xray_manifest.csv"
count = 0

with manifest_path.open("w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(
        csvfile,
        fieldnames=[
            "id", "image_file", "label", "labels", "source_url",
            "source_license", "rights_status", "width", "height"
        ],
    )
    writer.writeheader()

    for parquet_file in sorted(ROOT.glob("*.parquet")):
        table = pq.read_table(parquet_file)

        for row in table.to_pylist():
            if row.get("modality") != "xray":
                continue

            image_bytes = (row.get("image") or {}).get("bytes")
            if not image_bytes:
                continue

            image_id = str(row.get("id") or hashlib.sha256(image_bytes).hexdigest()[:20])
            safe_id = "".join(c for c in image_id if c.isalnum() or c in "-_")
            image_name = f"{safe_id}.png"
            image_path = IMAGES / image_name

            # Validate the image before saving it.
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                width, height = image.size
                image.save(image_path, format="PNG")

            writer.writerow({
                "id": image_id,
                "image_file": image_name,
                "label": row.get("primary_label") or "",
                "labels": "|".join(row.get("labels") or []),
                "source_url": row.get("source_url") or "",
                "source_license": row.get("source_license") or "",
                "rights_status": row.get("rights_status") or "",
                "width": width,
                "height": height,
            })
            count += 1

print(f"Prepared {count} X-ray images.")
print(f"Images: {IMAGES}")
print(f"Manifest: {manifest_path}")
