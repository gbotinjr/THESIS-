"""Test a trained model on NEW images it has never seen (does NOT train or change the model).
Put images in dataset/extra_test/<class>/  e.g. dataset/extra_test/asphalt_dry/ and dataset/extra_test/concrete_dry/
(folder names must match labels.txt). Writes only to results/extra_*, never touches results/misclassified.

Usage:  python 07_test_extra_images.py             -> uses model/model.tflite + model/labels.txt
        python 07_test_extra_images.py model_v2    -> uses a backed-up model folder (e.g. the best run, before the shadow experiment)

Images are loaded EXACTLY like in training (decode -> bilinear resize to 224x224 -> raw float32 0-255).
Training-only augmentation (blur, noise, grayscale, brightness/contrast, shadows) is deliberately NOT applied at test time.
Prints: how many were called each class, exact-class accuracy, dry-vs-wet accuracy, and a 95% interval."""
import csv, shutil, sys
import numpy as np, tensorflow as tf
from pathlib import Path

MODEL_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("model")
EXTRA, OUT = Path("dataset/extra_test"), Path("results/extra_misclassified")
EXT = {".jpg", ".jpeg", ".png", ".bmp"}
labels = (MODEL_DIR / "labels.txt").read_text().splitlines()
it = tf.lite.Interpreter(model_path=str(MODEL_DIR / "model.tflite")); it.allocate_tensors()
inp, out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
print(f"Using model folder: {MODEL_DIR}   classes: {labels}")

def load_like_training(path):
    """Load an image EXACTLY the way training did (image_dataset_from_directory): decode -> bilinear resize to 224x224."""
    img = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224))              # bilinear, same as training
    return np.expand_dims(img.numpy().astype(np.float32), 0)

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)

if not EXTRA.is_dir():
    raise SystemExit(f"Folder {EXTRA} not found. Create it with one subfolder per class, e.g. {EXTRA}\\asphalt_dry")
if OUT.exists():
    shutil.rmtree(OUT)
Path("results").mkdir(exist_ok=True)

rows = []
for cls_dir in sorted(p for p in EXTRA.iterdir() if p.is_dir()):
    cls = cls_dir.name
    if cls not in labels:
        print(f"[skip] {cls} is not in labels.txt: {labels}"); continue
    files = sorted(f for f in cls_dir.glob("*") if f.suffix.lower() in EXT)
    counts = {l: 0 for l in labels}
    for f in files:
        try:
            x = load_like_training(f)
        except Exception:
            print(f"[unreadable - skipped] {f}"); continue
        it.set_tensor(inp, x); it.invoke()
        probs = it.get_tensor(out)[0]; pred = labels[int(np.argmax(probs))]
        counts[pred] += 1
        rows.append([cls, f.name, pred, round(float(probs.max()), 3)])
        if pred != cls:
            dest = OUT / f"{cls}__predicted_{pred}"; dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest / f"{probs.max():.2f}_{f.name}")
    n = sum(counts.values()); ok = counts[cls]
    cond = cls.rsplit("_", 1)[1]
    ok_cond = sum(v for l, v in counts.items() if l.rsplit("_", 1)[1] == cond)
    lo, hi = wilson(ok, n)
    print(f"\n{cls}: {n} images")
    for l in labels:
        print(f"   called {l:<14}{counts[l]:>5}  ({100*counts[l]/max(n,1):.1f}%)")
    print(f"   exact-class accuracy : {ok}/{n} = {100*ok/max(n,1):.1f}%   (95% interval about {lo:.0f}-{hi:.0f}%)")
    print(f"   {cond} vs other condition: {ok_cond}/{n} = {100*ok_cond/max(n,1):.1f}%")

with open("results/extra_test_results.csv", "w", newline="") as fh:
    csv.writer(fh).writerows([["actual", "file", "predicted", "confidence"]] + rows)
print("\nSaved results/extra_test_results.csv; wrong images copied to results/extra_misclassified/")