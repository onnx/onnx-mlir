import numpy as np
import os
import sys
import tempfile
import torch
import inspect

from torch._inductor.codecache import (
    FxGraphCachePickler,
    FxGraphHashDetails,
)
from torch._subclasses.fake_tensor import FakeTensor

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


class config:
    cache_size = 3


global_session_cache = SessionCache(config.cache_size)


onnxmlir_counter = 0


# Backend function for torch.compile for onnx-mlir
def onnxmlir_backend(gm: torch.fx.GraphModule, *args, **kwargs):
    # Options provided at torch.compile will determine how the torch model
    # is exported, compiled and run.
    # The args and kwargs are inputs provided at inference, namely call to
    # forward()
    onnxmlir_options = kwargs.get("options")

    # Backend to export, compile and run inference of model with onnxmlir.
    def onnxmlir_forward_fn(*args, **kwargs):
        onnxmlirtorch_object = ONNXMLIRTorch(gm, options=onnxmlir_options)
        return onnxmlirtorch_object(*args)

    return onnxmlir_forward_fn


class OMFxGraphHashDetails:
    """
    This class includes information to hash a GraphModule so that we don't need
    to recompile the graph again.
    Information includes:
        - A GraphModule: a symbolic representation of the model,
        - Compilation options for onnx-mlir
    """

    def __init__(self, gm: torch.fx.GraphModule, compile_options) -> None:
        self.gm = gm
        self.compile_options = compile_options


def generate_hash_key(gm: torch.fx.GraphModule, compile_options) -> str:
    pickler = FxGraphCachePickler(gm)
    details = OMFxGraphHashDetails(gm, compile_options)
    return pickler.get_hash(details)


class ONNXMLIRTorch:
    def __init__(self, torch_model, **kwargs):
        global onnxmlir_counter
        onnxmlir_counter += 1
        # Pytorch model.
        self.torch_model = torch_model
        # Caching onnx_mlir sessions to avoid recompilation.
        self.session_cache = global_session_cache

        # Information for compiling and running an onnx model.
        self.workdir = tempfile.TemporaryDirectory()
        self.onnx_model = None
        self.default_model_name = "model"

        # Each onnx model has a unique tag.
        self.tag = onnxmlir_counter

        # Args passed to onnx-mlir.
        self.onnxmlir_kwargs = {}
        if kwargs["options"] is not None:
            for k, v in kwargs["options"].items():
                # compile_tag in onnx_mlir is a string.
                if k == "compile_tag":
                    self.onnxmlir_kwargs[k] = str(self.tag)
                self.onnxmlir_kwargs[k] = v

    def __call__(self, *example_inputs):
        return self.forward(*example_inputs)

    def forward(self, *example_inputs):
        cache_key = generate_hash_key(
            self.torch_model, self.onnxmlir_kwargs["compile_options"]
        )
        # Check whether there is any cached compiled model
        cached_session = self.session_cache.get(cache_key)

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
            file_index = self.session_cache.victim()

            # Remove the old .onnx and .so files.
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

            # Export the pytorch model to onnx.
            self.export_model_to_onnx(example_inputs)

            # Create a session for compiling and running the onnx model.
            sess = self.create_onnx_mlir_session()

            # Replace the victim cache entry
            self.session_cache.put(cache_key, (self.tag, sess))
        else:
            # Use the InferenceSession
            _, sess = cached_session

        # Run the inference.
        # onnx_mlir accepts numpy arrays as inputs and outputs.
        om_inputs = [
            arg.numpy() for arg in example_inputs if isinstance(arg, torch.Tensor)
        ]
        om_outputs = sess.run(om_inputs)
        return [torch.from_numpy(output) for output in om_outputs]

    def get_dynamic_shapes_for_export(self) -> dict[str, dict[int, str]]:
        """
        This computes a dictionary of dynamic shapes to be used in torch.export.
        """
        dynamic_shapes = {}
        for node in self.torch_model.graph.nodes:
            if node.op != "placeholder":
                continue
            input_name = node.target
            input_arg = node.meta["example_value"]
            # SymInts are not real inputs to the onnx model,
            # but we need to set it so that the export does not
            # claim about input mismatch.
            if isinstance(input_arg, torch.SymInt):
                dynamic_shapes[input_name] = None
                continue
            # Get dynamic dimensions from dynamic input tensors.
            dynamic_dims = {}
            for dim_idx, dim_size in enumerate(input_arg.shape):
                if isinstance(dim_size, torch.SymInt):
                    dynamic_dims[dim_idx] = str(dim_size)
            if dynamic_dims:
                dynamic_shapes[input_name] = dynamic_dims
        return dynamic_shapes

    def export_model_to_onnx(self, example_inputs):
        model_name = self.default_model_name + str(self.tag) + ".onnx"
        self.onnx_model = os.path.join(self.workdir.name, model_name)
        dynamic_shapes = self.get_dynamic_shapes_for_export()
        torch.onnx.export(
            self.torch_model,
            example_inputs,
            self.onnx_model,
            dynamic_shapes=dynamic_shapes,
        )

    def create_onnx_mlir_session(self):
        # Compile onnx model and hook with pyruntime
        return InferenceSession(
            self.onnx_model,
            temp_dir=self.workdir,
            **self.onnxmlir_kwargs,
        )


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


def interceptForward(model):
    model.register_forward_hook(print_parameters, with_kwargs=True)
