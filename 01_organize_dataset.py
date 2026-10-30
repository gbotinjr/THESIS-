"""Split dataset/raw/<surface>/<dry|wet>/ into dataset/split/{train,val,test}/<surface>_<condition>/ (70/20/10).
Only folders that actually contain images are used, so it works with 4 classes now and 8 later."""
import random, shutil
from pathlib import Path
from PIL import Image

SEED = 42
RAW, OUT = Path("dataset/raw"), Path("dataset/split")
EXT = {".jpg", ".jpeg", ".png", ".bmp"}
R_TRAIN, R_VAL = 0.70, 0.20          # remainder (10%) -> test
MAX_PER_CLASS = 150                  # randomly keep at most this many images per class (set None to use all)

def is_valid(f):
    """False for empty (0-byte), truncated or otherwise unreadable image files."""
    try:
        if f.stat().st_size == 0:
            return False
        with Image.open(f) as im:
            im.load()
        return True
    except Exception:
        return False

rng = random.Random(SEED)
classes = {}
for surface in sorted(p for p in RAW.iterdir() if p.is_dir()):
    for cond in ("dry", "wet"):
        d = surface / cond
        found = sorted(f for f in d.glob("*") if f.suffix.lower() in EXT) if d.is_dir() else []
        imgs = [f for f in found if is_valid(f)]
        for f in found:
            if f not in imgs:
                print(f"[BAD IMAGE - skipped] {f}")
        if imgs:
            if MAX_PER_CLASS and len(imgs) > MAX_PER_CLASS:
                print(f"[cap] {surface.name}_{cond}: {len(imgs)} images found, randomly keeping {MAX_PER_CLASS}")
                imgs = rng.sample(imgs, MAX_PER_CLASS)
            classes[f"{surface.name}_{cond}"] = imgs
        else:
            print(f"[skip] {d} has no images")

if OUT.exists():
    shutil.rmtree(OUT)

print(f"\n{'class':<18}{'total':>6}{'train':>7}{'val':>6}{'test':>6}")
for name, imgs in classes.items():
    rng.shuffle(imgs)
    n = len(imgs)
    n_tr, n_va = round(n * R_TRAIN), round(n * R_VAL)
    parts = {"train": imgs[:n_tr], "val": imgs[n_tr:n_tr + n_va], "test": imgs[n_tr + n_va:]}
    for split, files in parts.items():
        dest = OUT / split / name
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / f.name)
    print(f"{name:<18}{n:>6}{len(parts['train']):>7}{len(parts['val']):>6}{len(parts['test']):>6}")
print(f"\n{len(classes)} classes written to {OUT}")
