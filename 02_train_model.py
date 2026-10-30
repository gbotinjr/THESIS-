"""Train MobileNetV3-Small (ImageNet weights) in two phases: (1) train the new head, (2) fine-tune top layers.
Preprocessing (0-255 -> model scale) is built into MobileNetV3, so the model takes RAW float32 pixels 0-255."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

IMG, BATCH, SEED = 224, 16, 42
EPOCHS_HEAD, EPOCHS_FT, UNFREEZE_LAST = 15, 25, 30
GRAY_PROB = 0.15   # share of TRAINING images randomly turned to grayscale (0.0 = off, for the baseline run)
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

# Augmentation: training data only (kept out of the model so the exported TFLite stays clean)
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
    tf.keras.layers.RandomContrast(0.15),
])

def random_gray(x, y):
    """Randomly convert some training images to grayscale (kept as 3 channels, still 0-255).
    Training only: the exported model and the Pi code still take normal RGB 0-255 images."""
    gray = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x))
    use = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) < GRAY_PROB
    return tf.where(use, gray, x), y

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
if GRAY_PROB > 0:
    train_ds = train_ds.map(random_gray, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

base = tf.keras.applications.MobileNetV3Small(
    input_shape=(IMG, IMG, 3), include_top=False, weights="imagenet",
    pooling="avg", include_preprocessing=True)
base.trainable = False
inputs = tf.keras.Input((IMG, IMG, 3))
x = base(inputs, training=False)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

def compile_(lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])

def callbacks():
    return [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)]

print("\n=== Phase 1: training classifier head ===")
compile_(1e-3)
h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=callbacks())

print("\n=== Phase 2: fine-tuning top layers ===")
base.trainable = True
for layer in base.layers[:-UNFREEZE_LAST]:
    layer.trainable = False
for layer in base.layers:                       # keep BatchNorm statistics frozen
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
compile_(3e-5)
h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, callbacks=callbacks())

model.save("model/model.keras")
print("\nSaved model/model.keras and model/labels.txt")

acc = h1.history["accuracy"] + h2.history["accuracy"]
vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
loss = h1.history["loss"] + h2.history["loss"]
vloss = h1.history["val_loss"] + h2.history["val_loss"]
split = len(h1.history["accuracy"])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, (t, v, name) in zip(ax, [(acc, vacc, "Accuracy"), (loss, vloss, "Loss")]):
    a.plot(t, label="train"); a.plot(v, label="validation")
    a.axvline(split - 0.5, ls="--", c="gray", label="fine-tune starts")
    a.set_title(name); a.set_xlabel("epoch"); a.legend()
plt.tight_layout(); plt.savefig("results/training_curves.png", dpi=150)
print(f"Best validation accuracy: {max(vacc):.4f}")
