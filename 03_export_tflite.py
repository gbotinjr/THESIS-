"""Convert model/model.keras -> model/model.tflite and sanity-check it against the Keras model."""
import sys
import numpy as np, tensorflow as tf
from pathlib import Path

model = tf.keras.models.load_model("model/model.keras")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
if "--quant" in sys.argv:                       # optional: smaller/faster, tiny accuracy risk
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
Path("model/model.tflite").write_bytes(conv.convert())
print("Saved model/model.tflite  (%.2f MB)" % (Path("model/model.tflite").stat().st_size / 1e6))

# sanity check: same prediction as Keras on a random test image
img_path = next(Path("dataset/split/test").rglob("*.*"))
x = np.expand_dims(np.array(tf.keras.utils.load_img(img_path, target_size=(224, 224)), dtype=np.float32), 0)
it = tf.lite.Interpreter(model_path="model/model.tflite"); it.allocate_tensors()
it.set_tensor(it.get_input_details()[0]["index"], x); it.invoke()
p_tflite = it.get_tensor(it.get_output_details()[0]["index"])[0]
p_keras = model.predict(x, verbose=0)[0]
print("max abs difference Keras vs TFLite:", float(np.abs(p_tflite - p_keras).max()))
