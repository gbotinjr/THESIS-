"""Copy every wrongly predicted test image into results/misclassified/<actual>__predicted_<pred>/ so you can look at them."""
import shutil
import numpy as np, tensorflow as tf
from pathlib import Path

def load_like_training(path):
    """Load an image EXACTLY the way training did (image_dataset_from_directory): decode -> bilinear resize to 224x224."""
    img = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224))              # bilinear, same as training
    return np.expand_dims(img.numpy().astype(np.float32), 0)

labels = Path("model/labels.txt").read_text().splitlines()
it = tf.lite.Interpreter(model_path="model/model.tflite"); it.allocate_tensors()
inp, out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]

OUT = Path("results/misclassified")
if OUT.exists():
    shutil.rmtree(OUT)
wrong = 0
for cls in labels:
    for f in sorted((Path("dataset/split/test") / cls).glob("*.*")):
        x = load_like_training(f)
        it.set_tensor(inp, x); it.invoke()
        probs = it.get_tensor(out)[0]
        pred = labels[int(np.argmax(probs))]
        if pred != cls:
            wrong += 1
            dest = OUT / f"{cls}__predicted_{pred}"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest / f"{probs.max():.2f}_{f.name}")
            print(f"{f.name:<35} actual={cls:<14} predicted={pred:<14} confidence={probs.max():.2f}")
print(f"\n{wrong} wrong. Open the folder: explorer results\\misclassified")