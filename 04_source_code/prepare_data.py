import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


NAMES = ["mouse", "keyboard", "laptop", "headphones"]


def convert_label(source: Path) -> tuple[list[str], list[int], bool]:
    lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    converted, classes = [], []
    for line in lines:
        fields = line.split()
        old_id = int(fields[0])
        if old_id == 3:  # 删除原水杯类别
            continue
        new_id = 3 if old_id == 4 else old_id
        converted.append(" ".join([str(new_id), *fields[1:]]))
        classes.append(new_id)
    return converted, classes, bool(lines)


def public_split(image_id: str) -> str:
    value = int(hashlib.sha256(image_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "val" if value < 90 else "test"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    old_root = root / "训练前数据整理" / "训练就绪数据集"
    public_root = root / "公开数据扩充_v2" / "openimages_v7"
    output = root / "训练前数据整理" / "四分类训练集"

    if output.exists():
        shutil.rmtree(output)
    for split in ("train", "val", "test"):
        (output / "dataset" / "images" / split).mkdir(parents=True)
        (output / "dataset" / "labels" / split).mkdir(parents=True)

    records, seen_hashes = [], set()

    def add(image: Path, label: Path, split: str, name: str, source: str, detail: str) -> None:
        converted, classes, had_boxes = convert_label(label)
        if had_boxes and not converted:  # 排除仅含水杯的图片
            return
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest in seen_hashes:
            return
        seen_hashes.add(digest)
        target_image = output / "dataset" / "images" / split / name
        target_label = output / "dataset" / "labels" / split / f"{Path(name).stem}.txt"
        shutil.copy2(image, target_image)
        target_label.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
        records.append({
            "file": name,
            "split": split,
            "source": source,
            "source_detail": detail,
            "classes": ";".join(NAMES[i] for i in sorted(set(classes))),
            "box_count": len(classes),
            "sha256": digest,
        })

    with (old_root / "MANIFEST.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            split, filename = row["split"], row["file"]
            add(
                old_root / "dataset" / "images" / split / filename,
                old_root / "dataset" / "labels" / split / f"{Path(filename).stem}.txt",
                split,
                f"OLD_{filename}",
                row["source"],
                row["source_detail"],
            )

    with (public_root / "PUBLIC_MANIFEST.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            image_id = row["openimages_image_id"]
            add(
                public_root / Path(row["local_image"]),
                public_root / Path(row["local_label"]),
                public_split(image_id),
                f"OI7_{image_id}.jpg",
                "openimages_v7",
                row["original_landing_url"],
            )

    with (output / "MANIFEST.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    yaml = [
        f"path: {str((output / 'dataset').resolve()).replace(chr(92), '/')}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(NAMES)],
    ]
    (output / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    images = Counter(row["split"] for row in records)
    boxes = {split: Counter() for split in ("train", "val", "test")}
    for row in records:
        label = output / "dataset" / "labels" / row["split"] / f"{Path(row['file']).stem}.txt"
        boxes[row["split"]].update(int(line.split()[0]) for line in label.read_text().splitlines())
    summary = {
        "classes": NAMES,
        "images": dict(images),
        "boxes": {split: {NAMES[i]: boxes[split][i] for i in range(4)} for split in boxes},
        "total_images": len(records),
        "total_boxes": sum(sum(counts.values()) for counts in boxes.values()),
        "exact_duplicates": len(records) - len(seen_hashes),
    }
    (output / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
