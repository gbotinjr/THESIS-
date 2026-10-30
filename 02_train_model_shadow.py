"""Train MobileNetV3-Small (ImageNet weights) in two phases: (1) train the new head, (2) fine-tune top layers.
Preprocessing (0-255 -> model scale) is built into MobileNetV3, so the model takes RAW float32 RGB pixels 0-255.

This is your updated script PLUS one new training-only augmentation: RANDOM SHADOWS (see SHADOW_* settings).
Why: dry concrete with shadows, sun patches, cracks or stains was being read as "wet" because the wet photos are darker.
Randomly darkening parts of images of EVERY class breaks the shortcut "dark patch = wet".
Set SHADOW_PROB = 0.0 to turn the shadows off (results will be close to, but not bit-for-bit identical to, your previous run,
because training uses randomness). The exported TFLite model and the Pi code do not change."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path

# ----------------------------- settings -----------------------------
IMG, BATCH, SEED = 224, 16, 42
EPOCHS_HEAD, EPOCHS_FT, UNFREEZE_LAST = 15, 25, 30
WEIGHTS = "imagenet"

SHADOW_PROB = 0.40        # NEW: share of training images that get a random shadow     (0.0 = off)
SHADOW_MIN, SHADOW_MAX = 0.25, 0.60   # NEW: how much the shadowed side is darkened (0.25 = 25% darker, 0.60 = 60% darker)
SHADOW_SOFTNESS = 10.0    # NEW: higher = sharper shadow edge, lower = softer edge

GRAY_PROB = 0.15          # share of training images turned to grayscale        (0.0 = off)
BLUR_PROB = 0.25          # share of training images randomly blurred            (0.0 = off)  -> less reliance on sharpness
BLUR_SIGMAS = (0.6, 1.0, 1.5)   # blur strengths picked at random per batch
NOISE_STD = 4.0           # gaussian pixel noise (0-255 scale), 0.0 = off        -> less reliance on fine grain
BRIGHTNESS = 0.25         # +/- brightness jitter                                -> less reliance on lighting
CONTRAST = 0.25           # +/- contrast jitter
USE_CLASS_WEIGHTS = True  # balance classes so the model is not pulled toward the biggest/"catch-all" class
LABEL_SMOOTHING = 0.05    # softer labels: less overconfident wrong answers (0.0 = off)
MONITOR = "val_loss"      # early stopping signal (steadier than val_accuracy on a small validation set)
PATIENCE = 8
# ---------------------------------------------------------------------

Path("model").mkdir(exist_ok=True); Path("results").mkdir(exist_ok=True)
tf.keras.utils.set_random_seed(SEED)

def load(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        f"dataset/split/{split}", image_size=(IMG, IMG), batch_size=BATCH,
        label_mode="categorical", shuffle=shuffle, seed=SEED)

train_ds, val_ds = load("train", True), load("val", False)
class_names = train_ds.class_names
n_classes = len(class_names)
print("Classes:", class_names)
Path("model/labels.txt").write_text("\n".join(class_names))

# class weights from the number of training images per class
counts = np.array([len(list((Path("dataset/split/train") / c).glob("*.*"))) for c in class_names], dtype=float)
class_weight = {i: float(counts.sum() / (n_classes * counts[i])) for i in range(n_classes)} if USE_CLASS_WEIGHTS else None
print("Training images per class:", dict(zip(class_names, counts.astype(int))))
print("Class weights:", {class_names[i]: round(w, 2) for i, w in class_weight.items()} if class_weight else "off")

# ---------------- training-only augmentation ----------------
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(BRIGHTNESS, value_range=(0, 255)),
    tf.keras.layers.RandomContrast(CONTRAST),
])

def _pick(x, y_new, p):
    """For each image in the batch, use the changed version with probability p."""
    use = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) < p
    return tf.where(use, y_new, x)

def random_shadow(x, y):
    """Darken one side of a randomly placed, randomly angled soft-edged line, for a random share of the images."""
    b = tf.shape(x)[0]
    yy, xx = tf.meshgrid(tf.linspace(-1.0, 1.0, IMG), tf.linspace(-1.0, 1.0, IMG), indexing="ij")   # (H, W)
    theta = tf.random.uniform([b, 1, 1], 0.0, 2.0 * np.pi)                      # line direction
    offset = tf.random.uniform([b, 1, 1], -0.5, 0.5)                            # where the line sits
    dist = xx[None] * tf.cos(theta) + yy[None] * tf.sin(theta) + offset         # (B, H, W) signed distance to the line
    mask = tf.sigmoid(dist * SHADOW_SOFTNESS)                                   # 0 on the lit side -> 1 on the shadow side
    strength = tf.random.uniform([b, 1, 1], SHADOW_MIN, SHADOW_MAX)             # how dark the shadow is
    use = tf.cast(tf.random.uniform([b, 1, 1]) < SHADOW_PROB, tf.float32)       # which images get a shadow
    factor = 1.0 - use * strength * mask                                        # (B, H, W), 1.0 = unchanged
    return x * factor[..., None], y

def random_gray(x, y):
    gray = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x))
    return _pick(x, gray, GRAY_PROB), y

def _gauss_kernel(sigma, size=7):
    ax = tf.range(-(size // 2), size // 2 + 1, dtype=tf.float32)
    g = tf.exp(-(ax ** 2) / (2.0 * sigma ** 2)); g = g / tf.reduce_sum(g)
    k = g[:, None] * g[None, :]
    return tf.tile(k[:, :, None, None], [1, 1, 3, 1])           # depthwise kernel (7,7,3,1)

def random_blur(x, y):
    sigma = tf.gather(tf.constant(BLUR_SIGMAS, tf.float32), tf.random.uniform([], 0, len(BLUR_SIGMAS), tf.int32))
    blurred = tf.nn.depthwise_conv2d(x, _gauss_kernel(sigma), [1, 1, 1, 1], "SAME")
    return _pick(x, blurred, BLUR_PROB), y

def add_noise(x, y):
    return tf.clip_by_value(x + tf.random.normal(tf.shape(x), stddev=NOISE_STD), 0.0, 255.0), y

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
if SHADOW_PROB > 0: train_ds = train_ds.map(random_shadow, num_parallel_calls=AUTOTUNE)
if GRAY_PROB > 0:   train_ds = train_ds.map(random_gray, num_parallel_calls=AUTOTUNE)
if BLUR_PROB > 0:   train_ds = train_ds.map(random_blur, num_parallel_calls=AUTOTUNE)
if NOISE_STD > 0:   train_ds = train_ds.map(add_noise, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# save a picture of what the model actually trains on (check that the shadows look natural, not absurd)
try:
    imgs, _ = next(iter(train_ds))
    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for a, im in zip(axes.ravel(), imgs.numpy()[:8]):
        a.imshow(np.clip(im, 0, 255).astype("uint8")); a.axis("off")
    plt.tight_layout(); plt.savefig("results/augmentation_preview.png", dpi=110); plt.close(fig)
    print("Saved results/augmentation_preview.png (sample of augmented training images)")
except Exception as e:
    print("WARNING: could not save augmentation preview:", e)

# ---------------- model ----------------
base = tf.keras.applications.MobileNetV3Small(
    input_shape=(IMG, IMG, 3), include_top=False, weights=WEIGHTS,
    pooling="avg", include_preprocessing=True)
base.trainable = False
inputs = tf.keras.Input((IMG, IMG, 3))
x = base(inputs, training=False)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

def compile_(lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
                  metrics=["accuracy"])

def callbacks():
    return [tf.keras.callbacks.EarlyStopping(monitor=MONITOR, patience=PATIENCE, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)]

print("\n=== Phase 1: training classifier head ===")
compile_(1e-3)
h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=callbacks(), class_weight=class_weight)

print("\n=== Phase 2: fine-tuning top layers ===")
base.trainable = True
for layer in base.layers[:-UNFREEZE_LAST]:
    layer.trainable = False
for layer in base.layers:                       # keep BatchNorm statistics frozen
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
compile_(3e-5)
h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, callbacks=callbacks(), class_weight=class_weight)

model.save("model/model.keras")
print("\nSaved model/model.keras and model/labels.txt")

acc = h1.history["accuracy"] + h2.history["accuracy"]
vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
loss = h1.history["loss"] + h2.history["loss"]
vloss = h1.history["val_loss"] + h2.history["val_loss"]
split = len(h1.history["accuracy"])

# save the history first, so it is never lost even if plotting fails
Path("results/training_history.json").write_text(json.dumps(
    {"accuracy": acc, "val_accuracy": vacc, "loss": loss, "val_loss": vloss, "phase1_epochs": split,
     "phase2_epochs": len(acc) - split}, indent=1))
print(f"Epochs run: phase 1 = {split}, phase 2 = {len(acc) - split}, total = {len(acc)}")

try:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, (t, v, name) in zip(ax, [(acc, vacc, "Accuracy"), (loss, vloss, "Loss")]):
        a.plot(t, label="train"); a.plot(v, label="validation")
        a.axvline(split - 0.5, ls="--", c="gray", label="fine-tune starts")
        a.set_title(name); a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.savefig("results/training_curves.png", dpi=150)
except Exception as e:
    print("WARNING: could not save results/training_curves.png:", e,
          "\n         (the model is saved; close any image viewer showing that file and delete it, then re-run only if you need the plot)")
print(f"Best validation accuracy: {max(vacc):.4f}   lowest validation loss: {min(vloss):.4f}")
