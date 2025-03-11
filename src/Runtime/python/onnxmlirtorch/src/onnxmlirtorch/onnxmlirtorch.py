import numpy as np
import os
import sys
import tempfile
import torch

import onnxmlir
from .sessioncache import SessionCache


def compile(torch_model, **kwargs):
    return ONNXMLIRTorch(torch_model, **kwargs)


class config:
    cache_size = 3


class ONNXMLIRTorch:
    def __init__(self, torch_model, **kwargs):
        self.torch_model = torch_model
        self.kwargs = kwargs
        # Temporary directory
        self.workdir = tempfile.TemporaryDirectory()
        self.default_model_name = "model"
        self.sessionCache = SessionCache(config.cache_size)

    @staticmethod
    def compile(torch_model, **kwargs):
        return ONNXMLIRTorch(torch_model, **kwargs)

    def __call__(self, *args):

        # Convert the torch tensor to numpy tensor if needed
        np_args = [arg.numpy() for arg in args]

        # Get the shape of the input and convert it to string to be used as key
        input_shapes = str([arg.shape for arg in np_args])

        # Check whether there is any cached compiled model
        cached_session = self.sessionCache.get(input_shapes)

        if cached_session is None:
            # When there is no cached compiled lib, export the torch model
            # to an onnx model and compile it to a .so file.
            # Since the session is connected to a .so file, we have to make
            # sure that .so file exists with cached session.
            # In the meantime, we want keep a limited number of temporary files
            # for .onnx and .so file.
            # The solution is to name these file with suffix in [0, cache size)
            # The index for key in the dictionary may change. An extra index
            # value is store with session as the value in the cache.

            cache_index = self.sessionCache.victim()
            model_name = self.default_model_name + str(cache_index) + ".onnx"
            self.onnx_model = os.path.join(self.workdir.name, model_name)
            torch.onnx.export(self.torch_model, args, self.onnx_model)

            # Compile onnx model and hook with pyruntime
            self.sess = onnxmlir.InferenceSession(
                self.onnx_model, temp_dir=self.workdir, **self.kwargs
            )
            # Replace the victim cache entry
            self.sessionCache.put(input_shapes, (cache_index, self.sess))
        else:
            # Use the InferenceSession
            _, self.sess = cached_session

        # Run the inference
        outputs = self.sess.run(None, np_args)
        return [torch.from_numpy(output) for output in outputs]
