import os
import sys
import tempfile
import inspect
import logging
import types

import numpy as np
import torch
from torch._inductor.codecache import (
    FxGraphCachePickler,
    FxGraphHashDetails,
)
from torch._subclasses.fake_tensor import FakeTensor

from .onnxmlirdocker import InferenceSession
from .sessioncache import SessionCache, CacheValue

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

# Freeze the model so that parameters (weights and biases) in
# the forward function's arguments become constants in GraphModule.
# Alternative way is setting TORCHDYNAMO_PREPARE_FREEZING=1
# torch._dynamo.config.specialize_int = True
torch._dynamo.config.assume_static_by_default = False
# torch._dynamo.config.allow_ignore_mark_dynamic = True
# torch._dynamo.config.force_unspec_int_unbacked_size_like_on_torchrec_kjt = True
# torch._dynamo.config.allow_unspec_int_on_nn_module = True
# torch._dynamo.config.install_free_tensors = True


logger = logging.getLogger(__name__)


class ONNXMLIRConfig:
    cache_size = 3


# An instance to cache onnx_mlir session so that there is no need to recompile the same model.
global_session_cache = SessionCache(ONNXMLIRConfig.cache_size)


# Backend function for torch.compile.
def onnxmlir_backend(gm: torch.fx.GraphModule, *args, **kwargs):
    # Options provided at torch.compile will determine how the torch model
    # is exported, compiled and run.
    # The args and kwargs are inputs provided at inference, namely call to
    # forward().
    onnxmlir_options = kwargs.get("options")

    # Backend to export, compile and run inference of model with onnxmlir.
    def onnxmlir_forward_fn(*args, **kwargs):
        onnxmlirtorch_object = ONNXMLIRTorch(gm, *args, options=onnxmlir_options)
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
    def __init__(self, gm: torch.fx.GraphModule, *args, **kwargs):
        logger.debug(f"Original example_inputs in __init__: {args}")

        # Pytorch model.
        self.gm = gm
        logger.debug(f"Original graph module: {self.gm}")

        # Rewrite the graph for exporting to onnx.
        self.example_inputs_indices, _ = self.rewrite_gm_for_export(*args)
        logger.debug(f"Rewritten graph module: {self.gm}")

        # Information for compiling and running an onnx model.
        self.workdir = tempfile.TemporaryDirectory()
        self.onnx_model = None
        self.default_model_name = "model"

        # Generate an unique key from the graph module.
        self.cache_key = generate_hash_key(self.gm, kwargs["options"])
        logger.debug(f"Cache key: {self.cache_key}")
        # Check whether there is any cached compiled model.
        self.cached_session = global_session_cache.get(self.cache_key)
        if self.cached_session:
            self.example_inputs_indices = self.cached_session.example_inputs_indices
        logger.debug(f"Example inputs indices: {self.example_inputs_indices}")

        # Each onnx model has a unique tag.
        # Use the cache key as a tag when compiling the onnx model.
        self.tag = self.cache_key

        # Args passed to onnx-mlir.
        self.onnxmlir_kwargs = {"compile_tag": str(self.tag)}
        if kwargs["options"] is not None:
            for k, v in kwargs["options"].items():
                self.onnxmlir_kwargs[k] = v

    def __call__(self, *example_inputs):
        logger.debug(f"Original example_inputs in __call__: {example_inputs}")
        tensor_example_inputs = self.get_real_inputs(example_inputs)
        return self.forward(*tensor_example_inputs)

    def forward(self, *example_inputs):
        logger.debug(f"Inputs to forward: {example_inputs}")
        if self.cached_session is None:
            logger.info("Export and compile the model.")
            # When there is no cached compiled lib, export the torch model
            # to an onnx model and compile it to a .so file.
            # Since the session is connected to a .so file, we have to make
            # sure that .so file exists with cached session.
            # The number of .onnx and .so files gradually increases.
            # In the meantime, we want keep a limited number of temporary files
            # for .onnx and .so file.
            # The solution is to store the tuple of (tag, session) in the cache
            # When a cache entry becomes a victim, the corresponding files,
            # such as onnx model and .so are removed.
            file_index = global_session_cache.victim()

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

            # Export the graph module to onnx.
            self.export_gm_to_onnx(example_inputs)

            # Create a session for compiling and running the onnx model.
            sess = self.create_onnxmlir_session()

            # Replace the victim cache entry.
            cache_value = CacheValue(
                tag=self.tag,
                sess=sess,
                example_inputs_indices=self.example_inputs_indices,
            )
            global_session_cache.put(self.cache_key, cache_value)
        else:
            logger.info("Found the model in the cache. No recompilation.")
            # Use the InferenceSession in the cache.
            sess = self.cached_session.sess

        # onnx_mlir accepts numpy arrays as inputs and outputs.
        om_inputs = [
            arg.numpy() for arg in example_inputs if isinstance(arg, torch.Tensor)
        ]
        # Run the inference.
        logger.debug(f"onnx_mlir input sig: {sess.input_signature()}")
        logger.debug(f"onnx_mlir output sig: {sess.output_signature()}")
        om_outputs = sess.run(om_inputs)
        return [torch.from_numpy(output) for output in om_outputs]

    def get_real_inputs(self, example_inputs):
        tensor_real_inputs = []
        for i in self.example_inputs_indices:
            x = example_inputs[i]
            if isinstance(x, int):
                tensor_real_inputs.append(torch.tensor(x, dtype=torch.int64))
            elif isinstance(x, torch.Tensor):
                tensor_real_inputs.append(x)
            else:
                raise ValueError("Unsupported input type. Consider to support it")
        return tuple(tensor_real_inputs)

    def get_dynamic_shapes_for_export(self) -> ([str], dict[str, dict[int, str]]):
        """
        This computes a dictionary of dynamic shapes to be used in torch.export.
        """
        dynamic_shapes = {}
        input_names = []
        for node in self.gm.graph.nodes:
            if node.op == "output":
                # TODO explore node.args to build output_names
                continue
            if node.op != "placeholder":
                continue
            input_name = node.target
            input_arg = node.meta["example_value"]
            input_names.append(input_name)

            # SymInts are not real inputs to the onnx model
            # and Parameters are constants at inference time,
            # but we need to set them so that the export does not
            # complain about input name mismatch.
            if isinstance(input_arg, torch.SymInt) or isinstance(
                input_arg, torch.nn.Parameter
            ):
                dynamic_shapes[input_name] = None
                continue
            # Get dynamic dimensions from dynamic input tensors.
            dynamic_dims = {}
            for dim_idx, dim_size in enumerate(input_arg.shape):
                if isinstance(dim_size, torch.SymInt):
                    dynamic_dims[dim_idx] = "dim" + str(dim_size)
            if dynamic_dims:
                dynamic_shapes[input_name] = dynamic_dims
            else:
                dynamic_shapes[input_name] = None
        logger.debug(f"dynamic_shapes: {dynamic_shapes}")
        return input_names, dynamic_shapes

    def rewrite_gm_for_export(self, *example_inputs):
        example_inputs_indices, removed_example_inputs, constant_values = (
            self.extract_constants_from_example_inputs(example_inputs)
        )
        self.freeze_constants_with_values(constant_values)
        self.convert_symint_args_to_tensors(self.gm)
        return example_inputs_indices, removed_example_inputs

    def convert_symint_args_to_tensors(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        placeholders_to_replace = []

        # First pass: collect SymInt placeholders.
        for node in list(graph.nodes):
            if node.op == "placeholder" and node.type in [int, torch.SymInt]:
                new_name = f"{node.name}_tensor"
                if node.type is torch.SymInt:
                    value = int(node.meta["example_value"])
                else:
                    value = node.meta["example_value"]
                with graph.inserting_before(node):
                    new_node = graph.placeholder(new_name)
                    new_node.meta = {
                        "tensor_meta": {"shape": [1], "dtype": torch.int64},
                        "example_value": torch.tensor([value], dtype=torch.int64),
                    }
                    new_node.type = torch.Tensor
                placeholders_to_replace.append((node, new_node))

        # Second pass: replace uses with .item() calls.
        for old_node, new_node in placeholders_to_replace:
            for user in list(old_node.users):
                with graph.inserting_before(user):
                    item_node = graph.call_method("item", args=(new_node,))
                    user.replace_input_with(old_node, item_node)
            graph.erase_node(old_node)

        graph.lint()
        graph_module.recompile()
        return graph_module

    def extract_constants_from_example_inputs(self, example_inputs: tuple):
        graph = self.gm.graph
        placeholder_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        input_names = [n.name for n in placeholder_nodes]

        # Map input names to example values.
        name_to_value = dict(zip(input_names, example_inputs))

        # Detect constants using previous heuristic.
        constants = []
        name_to_use_nodes = {}
        for node in placeholder_nodes:
            uses = [n for n in graph.nodes if node in n.all_input_nodes]
            item_nodes = [
                n for n in uses if n.op == "call_method" and n.target == "item"
            ]
            if not item_nodes:
                continue
            item_node = item_nodes[0]
            item_uses = [n for n in graph.nodes if item_node in n.all_input_nodes]
            other_uses = [n for n in uses if n != item_node]
            if item_uses and not other_uses:
                constants.append(node.name)
                name_to_use_nodes[node.name] = item_nodes

        # Build constant_values dict.
        constant_values = {
            name: (name_to_value[name], name_to_use_nodes[name])
            for name in constants
            if name in name_to_value
        }

        example_inputs_indices = []
        removed_example_inputs_indices = []
        for i, name in enumerate(name_to_value.keys()):
            if name not in constants:
                example_inputs_indices.append(i)
            else:
                removed_example_inputs_indices.append(i)
        return example_inputs_indices, removed_example_inputs_indices, constant_values

    def freeze_constants_with_values(self, constant_values: dict):
        logger.debug(
            f"freeze_constants_with_values, constant_values: {constant_values}"
        )

        graph = self.gm.graph
        placeholder_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        name_to_node = {n.name: n for n in placeholder_nodes}

        for name, value_use_nodes in constant_values.items():
            value, use_nodes = value_use_nodes
            logger.debug(f"freeze_constants_with_values, {name}, {value}")
            if name not in name_to_node:
                continue
            node = name_to_node[name]

            # Register scalar or tensor
            if isinstance(value, torch.Tensor):
                self.gm.register_buffer(name, value)
            else:
                setattr(self.gm, name, value)

            # Insert get_attr node
            with graph.inserting_before(node):
                get_attr_node = graph.get_attr(name)

            # Replace all uses of the placeholder with get_attr
            for use_node in use_nodes:
                new_args = []
                for arg in use_node.args:
                    new_args.append(get_attr_node if arg == node else arg)
                use_node.args = tuple(new_args)

                # Optional: remove .item() calls if they follow the pattern
                # TODO(tung) only replace if not symint
                if use_node.op == "call_method" and use_node.target == "item":
                    logger.debug(
                        f"freeze_constants_with_values, replace {use_node} by {value}"
                    )
                    # Replace the .item() node with the scalar directly
                    scalar_value = (
                        value.item() if isinstance(value, torch.Tensor) else value
                    )
                    use_node.replace_all_uses_with(scalar_value)
                    graph.erase_node(use_node)
            logger.debug(f"freeze_constants_with_values, {name}, {value} END")

            graph.erase_node(node)

        graph.lint()
        self.gm.recompile()
        return self.gm

    def export_gm_to_onnx(self, example_inputs):
        model_name = self.default_model_name + str(self.tag) + ".onnx"
        self.onnx_model = os.path.join(self.workdir.name, model_name)
        input_names, dynamic_shapes = self.get_dynamic_shapes_for_export()
        torch.onnx.export(
            self.gm,
            example_inputs,
            self.onnx_model,
            input_names=input_names,
            dynamic_shapes=dynamic_shapes,
            external_data=False,
        )

    def create_onnxmlir_session(self) -> InferenceSession:
        # Return a session to compile and run the onnx model.
        # import shutil
        # shutil.copytree(self.workdir.name, '/home1/tung/dlc-backend-torch/debug')
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
