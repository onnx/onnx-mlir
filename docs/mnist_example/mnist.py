import numpy as np
from PyRuntime import ExecutionSession

# Load the model mnist.so compiled with onnx-mlir.
model = 'mnist.so'
session = ExecutionSession(model, "run_main_graph")
# Print the models input/output signature, for display.
print("input signature in json", session.input_signature())
print("output signature in json",session.output_signature())
# Create an input arbitrarily filled of 1.0 values.
input = np.full((1, 1, 28, 28), 1, np.dtype(np.float32))
# Run the model.
outputs = session.run([input])
# Analyze the output (first array in the list, of signature 1x10xf32).
prediction = outputs[0]
digit = -1
prob = 0.0
for i in range(0, 10):
    print("prediction ", i, "=", prediction[0, i])
    if prediction[0, i] > prob:
        digit = i
        prob = prediction[0, i]
# Print the value with the highest prediction (8 here).
print("The digit is", digit)
