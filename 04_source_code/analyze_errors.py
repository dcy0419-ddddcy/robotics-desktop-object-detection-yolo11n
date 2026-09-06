import csv
import json
import shutil
from collections import Counter
from pathlib import Path


NAMES = ["mouse", "keyboard", "laptop", "headphones"]


def load_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 5:
            boxes.append((int(fields[0]), *map(float, fields[1:5])))
    return boxes


def iou(a: tuple, b: tuple) -> float:
    _, ax, ay, aw, ah = a
    _, bx, by, bw, bh = b
    a1, a2, a3, a4 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    b1, b2, b3, b4 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    inter = max(0, min(a3, b3) - max(a1, b1)) * max(0, min(a4, b4) - max(a2, b2))
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    truth_root = root / "02_dataset" / "four_class_dataset" / "dataset" / "labels" / "test"
    pred_root = root / "05_results" / "predictions"
    output = root / "05_results" / "reproduced_error_analysis"
    output.mkdir(exist_ok=True)
    for old in output.glob("*.jpg"):
        old.unlink()

    totals, matched, rows = Counter(), Counter(), []
    for truth_file in sorted(truth_root.glob("*.txt")):
        truth = load_boxes(truth_file)
        pred = load_boxes(pred_root / "labels" / truth_file.name)
        totals.update(box[0] for box in truth)
        used, true_positive = set(), 0
        for gt in truth:
            candidates = [(iou(gt, box), index) for index, box in enumerate(pred)
                          if index not in used and box[0] == gt[0]]
            best_iou, best_index = max(candidates, default=(0, -1))
            if best_iou >= 0.5:
                used.add(best_index)
                matched[gt[0]] += 1
                true_positive += 1
        false_negative = len(truth) - true_positive
        false_positive = len(pred) - true_positive
        if false_negative or false_positive:
            rows.append({
                "image": f"{truth_file.stem}.jpg",
                "ground_truth": len(truth),
                "true_positive": true_positive,
                "false_negative": false_negative,
                "false_positive": false_positive,
                "error_score": false_negative * 2 + false_positive,
            })

    rows.sort(key=lambda row: (-row["error_score"], -row["false_negative"], row["image"]))
    with (output / "ERROR_CASES.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows[:12]:
        source = pred_root / row["image"]
        if source.exists():
            shutil.copy2(source, output / source.name)

    summary = {
        "iou_threshold": 0.5,
        "confidence_threshold": 0.25,
        "ground_truth": {NAMES[i]: totals[i] for i in range(4)},
        "matched": {NAMES[i]: matched[i] for i in range(4)},
        "recognition_rate": {NAMES[i]: round(matched[i] / totals[i], 4) for i in range(4)},
        "overall_recognition_rate": round(sum(matched.values()) / sum(totals.values()), 4),
        "images_with_errors": len(rows),
        "saved_typical_errors": min(12, len(rows)),
    }
    (output / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
