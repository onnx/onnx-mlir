import numpy as np
import os
import sys
import tempfile
import torch
import inspect

from .onnxmlirdocker import InferenceSession
from .sessioncache import SessionCache

"""
This file provides the utility to run inference of a torch model with onnx-mlir
compiler. There are two ways:

1. Add a wrapper to a torch model so that the onnx export, compile with 
onnx-mlir and run will happen automatically at inference of the model

Example code:
 
 # Assuem torch_model is the a torch model
 torch_model = ONNXLIRTorch(torch_model)
 results = torch_model(inputs) 

If user prefers the torch.compile style, the code could be:
 opt_model = onnxmlirtorch.compile(torch_model)
 results = opt_model(inputs)

The two format are identical in functionality.

2. Provide a customized backend, onnxmlir_backend, to torch.compile()
Example code:
 # Assuem torch_model is the a torch model
 opt_model = torch.compile(torch_model, backend=onnxmlirtorch.onnxmlir_backend)
 results = opt_model(inputs)

 You can provide options on how to compile the model to torch.compile.
Example code:
 myoptions={'compile_options' : '-O3',}
 opt_model = torch.compile(torch_model,
   backend = onnxmlirtorch.onnxmlir_backend,
   options = myoptions)
 results = opt_model(inputs)
 

Code structure:
Most of the functionality is implmented in class ONNXMLIRTorch. When an 
inference is called, the forward() function will export to the torch model
with the provided inputs to an onnx modeli (.onnx), compile the onnx model into
a library (.so), and run inference with the generated library. 
The ONNXMLIRTorch class also cache the existing session to reduce redundant
operation for possible reuse.
Components used by ONNXMLIRTorch:
 -- onnxmlirdocker: basic functionality of compile and run 
 -- SessionCache: cache for inteference session histroy

Current implementation checks the history of inference with only the shape of
the inputs, not the model itself. When torch.compile is used, the model from
different inference  may become difference due to the optimization based on
the inputs. Different ONNXMLIRTorch object will be created for each inference,
and there is no reuse with cache among them.
"""


# Alternative interface to minic the usage of torch.compile
def compile(torch_model, **kwargs):
    return ONNXMLIRTorch(torch_model, **kwargs)


def print_parameters(model, args, kwargs, outputs):
    print("------------ Begin ---------")
    fn = model.forward
    if fn is not None:
        signature = inspect.signature(fn)
        for param_name, param in signature.parameters.items():
            print(f"Parameter name: {param_name}")
    print(
        f"number of input parameters of forward call: args {len(args)}, kwargs {len(kwargs)}"
    )
    # Print out each parameter.
    # ToFix: save them into file
    print("args")
    for arg in args:
        print(arg)
    print("kwargs")
    for key, value in kwargs.items():
        print(f"{key} : {value}")
    print("------------ End ---------\n")


# Backend function for torch.compile for onnx-mlir
onnxmlir_counter = 1


def onnxmlir_backend(torch_model, *args, **kwargs):
    # Options provided at torch.compile will determine how the torch model
    # is exported, compiled and run.
    # The args and kwargs are inputs provided at inference, namely call to
    # forward()
    compile_options = kwargs.get("options")

    # Backend to export, compile and run inference of model with onnxmlir.
    def onnxmlir_forward_fn(*args, **kwargs):
        global onnxmlir_counter
        onnxmlir_counter += 1
        if compile_options is not None:
            onnxmlirtorchObject = ONNXMLIRTorch(
                torch_model, compile_tag=onnxmlir_counter, **compile_options
            )
        else:
            onnxmlirtorchObject = ONNXMLIRTorch(
                torch_model, compile_tag=onnxmlir_counter
            )
        return onnxmlirtorchObject(*args, **kwargs)

    return onnxmlir_forward_fn


def interceptForward(model):
    model.register_forward_hook(print_parameters, with_kwargs=True)


class config:
    cache_size = 3


class ONNXMLIRTorch:
    def __init__(self, torch_model, **kwargs):
        self.torch_model = torch_model
        # Temporary directory
        self.workdir = tempfile.TemporaryDirectory()
        self.default_model_name = "model"
        self.sessionCache = SessionCache(config.cache_size)
        if "compile_tag" in kwargs.keys():
            self.tag = kwargs["compile_tag"]
        else:
            self.tag = 0
        keys_to_remove = ["compile_tag"]
        new_kwargs = {k: v for k, v in kwargs.items() if k not in keys_to_remove}
        self.kwargs = new_kwargs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # Convert the torch tensor to numpy tensor if needed
        np_args = [arg.numpy() for arg in args]
        # Get the shape of the input and convert it to string to be used as key
        input_shapes = str([arg.shape for arg in args])

        # Check whether there is any cached compiled model
        cached_session = self.sessionCache.get(input_shapes)

        if cached_session is None:
            # When there is no cached compiled lib, export the torch model
            # to an onnx model and compile it to a .so file.
            # Since the session is connected to a .so file, we have to make
            # sure that .so file exists with cached session.
            # The .onnx and .so files as name with the tag, which continuous
            # increase
            # In the meantime, we want keep a limited number of temporary files
            # for .onnx and .so file.
            # The solution is to store the tuple of (tag, session) in the cache
            # When a cache entry becomes a victim, the corresponding files,
            # such as onnx model and .so are removed.

            file_index = self.sessionCache.victim()
            # Remove the
            old_onnx_file = os.path.join(
                self.workdir.name, self.default_model_name + str(file_index) + ".onnx"
            )
            if os.path.exists(old_onnx_file):
                os.remove(old_onnx_file)
            old_so_file = os.path.join(
                self.workdir.name, self.default_model_name + str(file_index) + ".so"
            )
            if os.path.exists(old_so_file):
                os.remove(old_so_file)

            self.tag += 1
            model_name = self.default_model_name + str(self.tag) + ".onnx"
            self.onnx_model = os.path.join(self.workdir.name, model_name)
            torch.onnx.export(self.torch_model, args, self.onnx_model)

            # Compile onnx model and hook with pyruntime
            sess = InferenceSession(
                self.onnx_model,
                temp_dir=self.workdir,
                compile_tag=str(self.tag),
                **self.kwargs,
            )
            # Replace the victim cache entry
            self.sessionCache.put(input_shapes, (self.tag, sess))
        else:
            # Use the InferenceSession
            _, sess = cached_session

        # Run the inference
        outputs = sess.run(None, np_args)
        return [torch.from_numpy(output) for output in outputs]
