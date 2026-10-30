"""Audit ONE class (default: concrete_dry). Shows which images the model gets wrong, how confident it was,
and whether the wrong ones differ from the right ones in simple image statistics. Run: python 06_audit_class.py [class_name]
Outputs: results/audit_<class>.csv and results/audit_<class>_wrong.png (contact sheet of the wrong images)."""
import csv, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path

TARGET = sys.argv[1] if len(sys.argv) > 1 else "concrete_dry"
labels = Path("model/labels.txt").read_text().splitlines()
if TARGET not in labels:
    sys.exit(f"'{TARGET}' is not one of {labels}")
SPLITS = ("train", "val", "test")
it = tf.lite.Interpreter(model_path="model/model.tflite"); it.allocate_tensors()
inp, out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]

def load_like_training(path):
    """Load an image EXACTLY the way training did (image_dataset_from_directory): decode -> bilinear resize to 224x224."""
    img = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224))
    return np.expand_dims(img.numpy().astype(np.float32), 0)

def stats(x):
    """x: 224x224x3 float 0-255 (the same pixels the model sees)."""
    g = x.mean(axis=2)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    mx, mn = x.max(axis=2), x.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
    return {"brightness": float(g.mean()), "contrast": float(g.std()), "sharpness": float(lap.var()),
            "saturation": float(sat.mean() * 100), "dark_pct": float((g < 60).mean() * 100)}
KEYS = ["brightness", "contrast", "sharpness", "saturation", "dark_pct"]

rows, images = [], {}
for cls in labels:
    for split in SPLITS:
        for f in sorted((Path("dataset/split") / split / cls).glob("*.*")):
            x = load_like_training(f)
            r = {"class": cls, "split": split, "file": f.name, **stats(x[0])}
            if cls == TARGET:
                it.set_tensor(inp, x); it.invoke()
                p = it.get_tensor(out)[0]
                r.update(predicted=labels[int(np.argmax(p))], confidence=float(p.max()),
                         **{f"p_{l}": float(v) for l, v in zip(labels, p)})
                images[(split, f.name)] = x[0].astype(np.uint8)
            rows.append(r)

tgt = [r for r in rows if r["class"] == TARGET]
for r in tgt:
    r["correct"] = r["predicted"] == TARGET
Path("results").mkdir(exist_ok=True)
with open(f"results/audit_{TARGET}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
    w.writeheader(); w.writerows(rows)

print(f"\nAUDIT of {TARGET}")
print(f"{'split':<8}{'images':>7}{'correct':>9}{'accuracy':>10}")
for split in SPLITS + ("val+test",):
    sel = [r for r in tgt if r["split"] in (split.split("+") if split == "val+test" else [split])]
    if sel:
        c = sum(r["correct"] for r in sel)
        print(f"{split:<8}{len(sel):>7}{c:>9}{100*c/len(sel):>9.1f}%")

wrong = sorted((r for r in tgt if not r["correct"]), key=lambda r: (r["split"] == "train", -r["confidence"]))
print(f"\nWRONG IMAGES ({len(wrong)}), most confident first within each group. A confident mistake on a TRAIN image often means a mislabeled photo.")
for r in wrong:
    print(f"  {r['split']:<6}{r['file']:<36}-> {r['predicted']:<14} conf={r['confidence']:.2f}")

print("\nAVERAGE IMAGE STATISTICS (same 224x224 pixels the model sees)")
print(f"{'group':<26}{'n':>5}" + "".join(f"{k:>12}" for k in KEYS))
groups = [(f"{TARGET} RIGHT", [r for r in tgt if r["correct"]]), (f"{TARGET} WRONG", wrong)]
groups += [(c, [r for r in rows if r["class"] == c]) for c in labels if c != TARGET]
for name, g in groups:
    if g:
        print(f"{name:<26}{len(g):>5}" + "".join(f"{np.mean([r[k] for r in g]):>12.1f}" for k in KEYS))

show = wrong[:24]
if show:
    cols = 6; nrows = (len(show) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 2.6, nrows * 3.0), squeeze=False)
    for a in axes.ravel():
        a.axis("off")
    for a, r in zip(axes.ravel(), show):
        a.imshow(images[(r["split"], r["file"])])
        a.set_title(f"{r['split']}  {r['file'][-14:]}\n-> {r['predicted']} ({r['confidence']:.2f})", fontsize=7)
    plt.tight_layout(); plt.savefig(f"results/audit_{TARGET}_wrong.png", dpi=130)
    print(f"\nSaved results/audit_{TARGET}_wrong.png and results/audit_{TARGET}.csv")