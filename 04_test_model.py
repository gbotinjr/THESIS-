"""Evaluate model/model.tflite (the exact model that runs on the Pi) on dataset/split/test.
Produces: Table 2 (per class-condition), surface-type and dry/wet accuracy, confusion matrix, precision/recall/F1."""
import csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, tensorflow as tf
from pathlib import Path

def load_like_training(path):
    """Load an image EXACTLY the way training did (image_dataset_from_directory): decode -> bilinear resize to 224x224."""
    img = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224))              # bilinear, same as training
    return np.expand_dims(img.numpy().astype(np.float32), 0)

labels = Path("model/labels.txt").read_text().splitlines()
idx = {n: i for i, n in enumerate(labels)}
it = tf.lite.Interpreter(model_path="model/model.tflite"); it.allocate_tensors()
inp, out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]

y_true, y_pred = [], []
for cls in labels:
    for f in sorted((Path("dataset/split/test") / cls).glob("*.*")):
        x = load_like_training(f)
        it.set_tensor(inp, x); it.invoke()
        y_true.append(idx[cls]); y_pred.append(int(np.argmax(it.get_tensor(out)[0])))
y_true, y_pred = np.array(y_true), np.array(y_pred)
n = len(labels)
cm = np.zeros((n, n), int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

Path("results").mkdir(exist_ok=True)
print("\nTABLE 2 - Testing table")
print(f"{'Class':<18}{'Test':>6}{'Correct':>9}{'Accuracy':>10}")
rows = []
for i, c in enumerate(labels):
    tot, cor = cm[i].sum(), cm[i, i]
    rows.append([c, tot, cor, round(100 * cor / tot, 2)])
    print(f"{c:<18}{tot:>6}{cor:>9}{100*cor/tot:>9.2f}%")
overall = 100 * np.trace(cm) / cm.sum()
print(f"{'TOTAL':<18}{cm.sum():>6}{np.trace(cm):>9}{overall:>9.2f}%")
rows.append(["TOTAL", int(cm.sum()), int(np.trace(cm)), round(overall, 2)])
with open("results/table2.csv", "w", newline="") as f:
    csv.writer(f).writerows([["class", "test_images", "correct", "accuracy_pct"]] + rows)

# accuracy by surface type only, and by condition only
surf = [l.rsplit("_", 1)[0] for l in labels]; cond = [l.rsplit("_", 1)[1] for l in labels]
def grouped(keys, title):
    print(f"\nAccuracy by {title} (both predictions must match on this attribute)")
    for k in sorted(set(keys)):
        m = np.array([keys[t] == k for t in y_true])
        ok = np.array([keys[t] == keys[p] for t, p in zip(y_true, y_pred)])[m]
        print(f"  {k:<10}{100*ok.mean():>7.2f}%  ({ok.sum()}/{len(ok)})")
grouped(surf, "surface type"); grouped(cond, "condition (dry/wet)")

print(f"\n{'Class':<18}{'Precision':>10}{'Recall':>8}{'F1':>8}")
for i, c in enumerate(labels):
    tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i].sum() - cm[i, i]
    pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
    print(f"{c:<18}{pr:>10.3f}{rc:>8.3f}{f1:>8.3f}")

fig, ax = plt.subplots(figsize=(1.1 * n + 3, 1.1 * n + 2))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
for i in range(n):
    for j in range(n):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=150)
print("\nSaved results/table2.csv and results/confusion_matrix.png")