import keras
import numpy as np

model=keras.models.load_model("nn_model.keras")
# ValueError: The model provided has never called. It must be called at least once before export.
# we need dummy input because of the error above
dummy_input=np.zeros((1,10000), dtype=np.float32)
model(dummy_input)

model.export("NN_model.onnx", format="onnx")