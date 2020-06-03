import onnx
import os
from onnx import optimizer

# this file to remove the warnings when using the ultra-light models.

listdir = os.listdir()
for file in listdir:
    if file.endswith('.onnx'):
        onnx_model = onnx.load(file)
        passes = ["extract_constant_to_initializer", "eliminate_unused_initializer",
                  "eliminate_deadend", "fuse_bn_into_conv"]
        optimized_model = optimizer.optimize(onnx_model, passes)
        onnx.save(optimized_model, 'win_fixed_'+file)

