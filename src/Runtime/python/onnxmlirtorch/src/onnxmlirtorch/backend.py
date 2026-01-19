# SPDX-License-Identifier: Apache-2.0

##################### backend.py *******########################################
#
# Copyright 2025 The IBM Research Authors.
#
################################################################################
#
# This file defines an onnx-mlir backend for torch.compile().
#
################################################################################

import io
import os
import sys
import tempfile
import time
import inspect
import logging
import types
import functools
import pickle
import pickletools
from collections import deque

import numpy as np
import torch
from torch._inductor.codecache import (
    _ident,
    extract_tensor_metadata_for_cache_key,
    FxGraphCachePickler,
    FxGraphHashDetails,
    sha256_hash,
)
from torch._subclasses.fake_tensor import (
    FakeTensor,
)

from .onnxmlirdocker import InferenceSession
from .sessioncache import SessionCache, CacheValue
from . import config

"""
This file provides an onnx-mlir compiler backend for torch.compile().

The backend can be used by passing onnxmlir_backend to torch.compile():
- torch.compile(model, backend=onnxmlir_backend, ...)
or using "onnxmlir" as the backend name:
- torch.compile(model, backend="onnxmlir", ...)

Below is one example of running a bert model using onnx-mlir backend.
```python
import torch
import onnxmlirtorch
from transformers import AutoModel, AutoTokenizer

model_path = "ibm-granite/granite-embedding-30m-english"
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

om_options = {
    "compiler_image_name": None,
    "compile_options": "-O3",
    "compiler_path": "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
}
compiled_model = torch.compile(
    model,
    backend="onnxmlir",
    options=om_options,
)


inputs = tokenizer("AI is fascinating", return_tensors="pt")
with torch.no_grad():
    outputs = compiled_model(**inputs)
```
"""


logger = logging.getLogger(__name__)

# An instance to cache onnx_mlir session so that there is no need to recompile the same model.
global_session_cache = SessionCache(config.session_cache_limit)


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


class OMFxGraphCachePickler(FxGraphCachePickler):
    """
    A class to serialize a FxGraph for hashing.
    """

    def __init__(self, gm: torch.fx.GraphModule):
        super().__init__(gm)
        # pyrefly: ignore  # bad-override
        self.dispatch_table: dict
        self.dispatch_table.update(
            {
                FakeTensor: functools.partial(self._reduce_fake_tensor),
                torch.Tensor: functools.partial(self._reduce_tensor),
                torch.nn.parameter.Parameter: functools.partial(self._reduce_tensor),
                torch.SymInt: functools.partial(self._reduce_symint),
            }
        )

    def _reduce_tensor(self, tensor):
        """
        Reduce the tensor to a stable key for caching.
        """
        metadata = extract_tensor_metadata_for_cache_key(tensor)
        return (_ident, (metadata,))


class OMFxGraphHashDetails(FxGraphHashDetails):
    """
    A class to capture all the details relevant to computing a safe and stable cache key.
    Information includes:
        - A GraphModule: a symbolic representation of the model,
        - Compilation options for onnx-mlir
    """

    def __init__(self, gm: torch.fx.GraphModule, compile_options) -> None:
        self.gm = gm
        self.compile_options = compile_options


def generate_hash_key(
    gm: torch.fx.GraphModule, compile_options, use_lightweight_hashing=True
) -> str:
    start = time.perf_counter()
    if use_lightweight_hashing:
        # Hash the graph module.
        # Touch the code to materialize.
        _ = gm.code

        # Generate a unique string to represent the graph module.
        graph_info = []
        placeholder_counter = 0
        dim_counter = 0
        dim_dict = {}
        for node in gm.graph.nodes:
            node_info = []
            # Use stable names for placeholders and symbolic dimensions.
            if node.op == "placeholder":
                if "example_value" in node.meta and isinstance(
                    node.meta["example_value"], torch.Tensor
                ):
                    shape = []
                    for d in node.meta["example_value"].shape:
                        s = str(d)
                        if isinstance(d, torch.SymInt):
                            if s in dim_dict:
                                shape.append(dim_dict[s])
                            else:
                                dim_str = f"dim_{dim_counter}"
                                dim_dict[s] = dim_str
                                dim_counter += 1
                                shape.append(dim_str)
                        else:
                            shape.append(s)
                    shape_str = ",".join(shape)
                    node_info.append(
                        f"om_placeholder_{placeholder_counter}_[{shape_str}]"
                    )
                else:
                    node_info.append(f"om_placeholder_{placeholder_counter}")
                placeholder_counter += 1
            else:
                node_info.append(f"{node.op}_{torch.typename(node.target)}")
                # Append information from input nodes.
                for inode in node._input_nodes.keys():
                    if inode.op == "get_attr":
                        try:
                            t = gm._parameters[inode.target]
                        except KeyError:
                            t = None
                        if t is not None and isinstance(t, torch.nn.Parameter):
                            sample_values = [
                                str(s)
                                for s in t.view(-1)[
                                    : config.sample_parameter_values_limit
                                ].tolist()
                            ]
                            sample_str = ".".join(sample_values)
                        else:
                            sample_str = "."
                        node_info.append(f"{inode.name}.{sample_str}")
                    else:
                        node_info.append(f"{inode.name}")
            graph_info.append(";".join(node_info))
        graph_str = " ".join(graph_info)
        graph_hash = sha256_hash(graph_str.encode())

        # Hash the options.
        with io.BytesIO() as stream:
            options_data = pickle.dumps(compile_options)
            options_opt = pickletools.optimize(options_data)
            options_hash = sha256_hash(options_opt)
        key = graph_hash + options_hash
    else:
        pickler = OMFxGraphCachePickler(gm)
        details = OMFxGraphHashDetails(gm, compile_options)
        key = pickler.get_hash(details)

    key = "_om_" + key
    logger.info(
        f"Creating a cache key took {(time.perf_counter() - start)*1000} ms: {key}"
    )
    return key


class ONNXMLIRTorch:
    def __init__(self, gm: torch.fx.GraphModule, *args, **kwargs):
        # Input graph module.
        self.gm = gm
        self.gm.eval()

        # Indice of the actual example inputs if the graph's signature is changed.
        self.example_inputs_indices = None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Original graph module {self.gm}")

        if self.use_eager_mode():
            if "om_example_inputs_indices" in self.gm.meta:
                self.example_inputs_indices = self.gm.meta["om_example_inputs_indices"]
            return

        # If the model was rewritten, the cache key was stored in "om_hash" in gm.meta.
        need_rewrite = False
        if "om_hash" not in self.gm.meta:
            # Rewrite the graph at the first time touching the graph.
            need_rewrite = True
            self.gm.meta["om_same_hash_counter"] = deque([])
        else:
            same_hash_counter = self.gm.meta["om_same_hash_counter"]
            same_hash_size = max(0, config.same_hash_size)
            if len(same_hash_counter) == same_hash_size and all(same_hash_counter):
                self.cache_key = self.gm.meta["om_hash"]
            else:
                self.cache_key = generate_hash_key(self.gm, kwargs["options"])
                if self.cache_key == self.gm.meta["om_hash"]:
                    same_hash_counter.append(True)
                else:
                    # Rewrite the graph if it was changed.
                    same_hash_counter.append(False)
                    need_rewrite = True
                while len(same_hash_counter) > same_hash_size:
                    if same_hash_counter:
                        same_hash_counter.popleft()
                self.gm.meta["om_same_hash_counter"] = same_hash_counter

        if need_rewrite:
            # Rewrite the graph for exporting to onnx.
            (
                self.example_inputs_indices,
                removed_example_inputs_indices,
                placeholders_to_replace,
            ) = self.rewrite_gm_for_export(*args)
            self.cache_key = generate_hash_key(self.gm, kwargs["options"])
            self.gm.meta["om_hash"] = self.cache_key

        # Cache the rewritten graph module.
        assert self.cache_key, "cache key does not exist"
        self.cached_session = global_session_cache.get(self.cache_key)
        if self.cached_session:
            self.example_inputs_indices = self.cached_session.example_inputs_indices

        # Touch the code to materialize before exporting.
        _ = self.gm.code

        # Information for compiling and running an onnx model.
        self.workdir = tempfile.TemporaryDirectory()
        self.onnx_model = None
        self.default_model_name = "model"
        # Each onnx model is assigned a unique tag.
        # Use the cache key as a tag when compiling the onnx model.
        self.tag = self.cache_key

        # Args passed to onnx-mlir.
        self.onnxmlir_kwargs = {"compile_tag": str(self.tag)}
        if kwargs["options"] is not None:
            for k, v in kwargs["options"].items():
                self.onnxmlir_kwargs[k] = v

    def __call__(self, *example_inputs):
        if self.use_eager_mode():
            fn = self.eager_forward
        else:
            fn = self.compile_forward
        tensor_example_inputs = self.get_tensor_example_inputs(example_inputs)
        return fn(*tensor_example_inputs)

    def eager_forward(self, *example_inputs):
        if "om_use_eager_mode" not in self.gm.meta:
            self.gm.meta["om_use_eager_mode"] = True
        if "om_example_inputs_indices" not in self.gm.meta:
            self.gm.meta["om_example_inputs_indices"] = self.example_inputs_indices

        logger.info("Use the eager mode to run the graph.")
        start = time.perf_counter()
        results = self.gm.forward(*example_inputs)
        logger.info(f"  torch took {(time.perf_counter() - start)*1000} ms")
        return results

    def compile_forward(self, *example_inputs):
        if self.cached_session is None:
            logger.info("Export and compile the model.")
            # When there is no cached compiled lib, export the torch model
            # to an onnx model and compile it to a .so file.
            # Since the session is connected to a .so file, we have to make
            # sure that .so file exists with cached session.
            # The number of .onnx and .so files gradually increases.
            # In the meantime, we want keep a limited number of temporary files
            # for .onnx and .so file.
            # The solution is to store the tag in the cache value.
            # When a cache entry becomes a victim, the corresponding files,
            # such as onnx model and .so are removed.
            tag_id = global_session_cache.victim()

            # Remove the old .onnx and .so files.
            self.cleanup_onnxmlir_files(tag_id)

            # Export the graph module to onnx.
            # If failed, use the graph as it is without compilation.
            if not self.export_gm_to_onnx(example_inputs):
                logger.info("Failed to export the model. Switch to the eager mode.")
                return self.eager_forward(*example_inputs)

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
        om_inputs = [arg.contiguous().numpy() for arg in example_inputs]
        # Run the inference.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"onnx_mlir input sig: {sess.input_signature()}")
            logger.debug(f"onnx_mlir output sig: {sess.output_signature()}")
        start = time.perf_counter()
        om_outputs = sess.run(om_inputs)
        logger.info(f"sess.run took {(time.perf_counter() - start)*1000} ms")
        return [torch.from_numpy(output) for output in om_outputs]

    def use_eager_mode(self):
        if "om_use_eager_mode" in self.gm.meta:
            return True
        return False

        # Detect unsupported ops.
        for n in self.gm.graph.nodes:
            # copy op
            if n.op == "call_method":
                if n.target == "copy_":
                    return True
            if n.op == "call_function":
                if n.target in (
                    torch.ops.aten.copy_.default,
                    torch.ops.aten.copy.default,
                    torch.ops.aten.sym_storage_offset.default,
                ):
                    return True

        return False

    def get_tensor_example_inputs(self, example_inputs):
        if self.example_inputs_indices is None:
            indices = range(len(example_inputs))
        else:
            indices = self.example_inputs_indices

        tensor_inputs = []
        for i in indices:
            x = example_inputs[i]
            if isinstance(x, int):
                tensor_inputs.append(torch.tensor(x, dtype=torch.int64))
            elif isinstance(x, torch.Tensor):
                tensor_inputs.append(x)
            else:
                raise ValueError("Unsupported input type. Consider to support it")
        return tuple(tensor_inputs)

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
                    if not str(dim_size).isdigit():
                        dynamic_dims[dim_idx] = "dim" + str(dim_size)
            if dynamic_dims:
                dynamic_shapes[input_name] = dynamic_dims
            else:
                dynamic_shapes[input_name] = None
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"dynamic_shapes: {dynamic_shapes}")
        return input_names, dynamic_shapes

    def rewrite_gm_for_export(self, *example_inputs):
        # Freeze scalar constant arguments that are typically parameters, e.g.,
        # epsilon value, from the config file of the model and they are constants.
        example_inputs_indices, removed_example_inputs, constant_values = (
            self.extract_scalar_constant_args(example_inputs)
        )

        self.freeze_scalar_constant_args(constant_values)
        # Since onnx does not support scalar inputs, symbolic integer arguments
        # are converted to tensor arguments.
        placeholders_to_replace = self.convert_symint_args_to_tensors()
        # After rewriting the argument list of the graph module, we maintain
        # a list of un-removed arguments that are used in forward for passing
        # correct example inputs to the rewritten graph module.
        return example_inputs_indices, removed_example_inputs, placeholders_to_replace

    def convert_symint_args_to_tensors(self):
        # Important note: do not cast SymInt to int by int(SymInt)
        # since that concretizes symbolic dimensions in related Tensors.

        graph = self.gm.graph
        placeholders_to_replace = []

        # First pass: collect SymInt placeholders.
        for node in graph.nodes:
            if node.op == "placeholder" and node.type in [int, torch.SymInt]:
                new_name = f"{node.name}_tensor"
                with graph.inserting_before(node):
                    new_node = graph.placeholder(new_name)
                    new_node.meta = node.meta
                    new_node.meta["tensor_meta"] = {"shape": [1], "dtype": torch.int64}
                    if node.type is int:
                        new_node.meta["example_value"] = torch.tensor(
                            [value], dtype=torch.int64
                        )
                    new_node.type = torch.Tensor
                placeholders_to_replace.append((node, new_node))

        # Second pass: replace uses with .item() calls.
        for old_node, new_node in placeholders_to_replace:
            for user in list(old_node.users):
                with graph.inserting_before(user):
                    item_node = graph.call_method("item", args=(new_node,))
                    user.replace_input_with(old_node, item_node)
            graph.erase_node(old_node)

        if placeholders_to_replace:
            self.gm.graph.lint()
            self.gm.recompile()

        return placeholders_to_replace

    def extract_scalar_constant_args(self, example_inputs: tuple):
        graph = self.gm.graph
        placeholder_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        input_names = [n.name for n in placeholder_nodes]

        # Map input names to example values.
        name_to_value = dict(zip(input_names, example_inputs))

        # Detect scalar constants by this pattern: placeholder -> .item().
        scalar_constants = []
        name_to_use_nodes = {}
        for node in placeholder_nodes:
            input_arg = node.meta["example_value"]
            # Not a tensor.
            if not isinstance(input_arg, torch.Tensor):
                continue
            # Not a scalar.
            if input_arg.ndim != 0:
                continue
            # Pattern: placeholder -> .item().
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
                scalar_constants.append(node.name)
                name_to_use_nodes[node.name] = item_nodes

        # Build constant_values dict.
        constant_values = {
            name: (name_to_value[name], name_to_use_nodes[name])
            for name in scalar_constants
            if name in name_to_value
        }

        # Keep lists of indices of example inputs that are removed and not removed.
        example_inputs_indices = []
        removed_example_inputs_indices = []
        for i, name in enumerate(name_to_value.keys()):
            if name not in scalar_constants:
                example_inputs_indices.append(i)
            else:
                removed_example_inputs_indices.append(i)
        return example_inputs_indices, removed_example_inputs_indices, constant_values

    def freeze_scalar_constant_args(self, constant_values: dict):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"freeze_scalar_constant_args, constant_values: {constant_values}"
            )

        if not constant_values:
            return

        graph = self.gm.graph
        placeholder_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        name_to_node = {n.name: n for n in placeholder_nodes}

        for name, value_use_nodes in constant_values.items():
            value, use_nodes = value_use_nodes
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"freeze_scalar_constant_args, {name}, {value}")
            if name not in name_to_node:
                continue
            node = name_to_node[name]

            # Register scalar or tensor.
            if isinstance(value, torch.Tensor):
                self.gm.register_buffer(name, value)
            else:
                setattr(self.gm, name, value)

            # Insert get_attr node.
            with graph.inserting_before(node):
                get_attr_node = graph.get_attr(name)

            # Replace all uses of the placeholder with get_attr.
            for use_node in use_nodes:
                new_args = []
                for arg in use_node.args:
                    new_args.append(get_attr_node if arg == node else arg)
                use_node.args = tuple(new_args)

                # Remove .item() calls if they follow the pattern.
                if use_node.op == "call_method" and use_node.target == "item":
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"freeze_scalar_constant_args, replace {use_node} by {value}"
                        )
                    # Replace the .item() node with the scalar directly.
                    scalar_value = (
                        value.item() if isinstance(value, torch.Tensor) else value
                    )
                    use_node.replace_all_uses_with(scalar_value)
                    graph.erase_node(use_node)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"freeze_scalar_constant_args, {name}, {value} END")

            graph.erase_node(node)

        graph.lint()
        self.gm.recompile()
        return self.gm

    def export_gm_to_onnx(self, example_inputs):
        model_name = self.default_model_name + str(self.tag) + ".onnx"
        self.onnx_model = os.path.join(self.workdir.name, model_name)
        input_names, dynamic_shapes = self.get_dynamic_shapes_for_export()

        succeeded = False
        try:
            torch.onnx.export(
                self.gm,
                example_inputs,
                self.onnx_model,
                input_names=input_names,
                dynamic_shapes=dynamic_shapes,
                dynamo=True,
                # dynamic_axes=dynamic_shapes,
                # dynamo=False,
            )
            succeeded = True
        except torch.onnx.errors.UnsupportedOperatorError as e:
            print("ONNX export unsupported:", e)
        except Exception as e:
            print("ONNX export failure:", e)

        return succeeded

    def create_onnxmlir_session(self) -> InferenceSession:
        # Return a session to compile and run the onnx model.
        return InferenceSession(
            self.onnx_model,
            temp_dir=self.workdir,
            **self.onnxmlir_kwargs,
        )

    def cleanup_onnxmlir_files(self, tag_id):
        base = os.path.join(self.workdir.name, self.default_model_name + str(tag_id))
        old_files = [base + ".onnx", base + ".constants.bin", base + ".so"]
        for f in old_files:
            if os.path.exists(f):
                os.remove(old_onnx_file)


# Alternative interface to minic the usage of torch.compile
def compile(torch_model, *args, **kwargs):
    return ONNXMLIRTorch(torch_model, *args, **kwargs)


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
