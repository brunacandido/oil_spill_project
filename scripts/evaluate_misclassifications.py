from pathlib import Path

# CHANGE THESE PATHS IF NEEDED
ground_truth_dir = Path("yolov5_data/labels/val")
prediction_dir = Path("yolov5_results/inference_final22/labels")

false_positives = []
false_negatives = []
true_positives = []

# Collect .txt files
gt_files = {f.stem: f for f in ground_truth_dir.glob("*.txt")}
pred_files = {f.stem: f for f in prediction_dir.glob("*.txt")}

# Combine all image names
all_keys = set(gt_files.keys()).union(pred_files.keys())

for key in sorted(all_keys):
    gt_path = gt_files.get(key)
    pred_path = pred_files.get(key)

    gt_has_box = gt_path and gt_path.read_text().strip() != ""
    pred_has_box = pred_path and pred_path.read_text().strip() != ""

    if pred_has_box and not gt_has_box:
        false_positives.append(key)
    elif gt_has_box and not pred_has_box:
        false_negatives.append(key)
    elif gt_has_box and pred_has_box:
        true_positives.append(key)

# Print results
print("\nFalse Positives:")
for name in false_positives:
    print(f"  {name}.jpg")

print("\nFalse Negatives:")
for name in false_negatives:
    print(f"  {name}.jpg")

print("\nTrue Positives:")
for name in true_positives:
    print(f"  {name}.jpg")
