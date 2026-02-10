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
from torch.export import Dim
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
from . import config, fx_utils

"""
This file provides an onnx-mlir compiler backend for torch.compile().

The backend can be used by passing onnxmlir_backend to torch.compile():
- torch.compile(model, backend=onnxmlir_backend, ...)
or using "onnxmlir" as the backend name:
- torch.compile(model, backend="onnxmlir", ...)

Below is one example of running a bert model using onnx-mlir backend.
```python
import torch
import torch_onnxmlir
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

global_uncompilable_graphs = set()


def has_unsupported_onnx_ops(gm: torch.fx.GraphModule):
    # Detect unsupported ops. Add unsupported ops here.
    return False


def eager_forward_fn(gm: torch.fx.GraphModule):
    stable_gm = torch.fx.GraphModule(gm, gm.graph)
    stable_gm.eval()

    # Ensure we don't re-enter Dynamo.
    @torch._dynamo.disable
    def run(*args):
        with torch.no_grad():
            start = time.perf_counter()
            results = stable_gm(*args)
            logger.info(f"Eager mode took {(time.perf_counter() - start)*1000} ms")
            return results

    return run


# Backend function for torch.compile.
def onnxmlir_backend(gm: torch.fx.GraphModule, *args, **kwargs):
    # Switch back to the eager mode if the graph has unsupported onnx ops.
    if has_unsupported_onnx_ops(gm):
        return eager_forward_fn(gm)

    # Options provided at torch.compile will determine how the torch model
    # is exported, compiled and run.
    # The args and kwargs are inputs provided at inference, namely call to
    # forward().
    onnxmlir_options = kwargs.get("options")

    # Backend to export, compile and run inference of model with onnxmlir.
    def onnxmlir_forward_fn(*args, **kwargs):
        torch_onnxmlir_object = TorchONNXMLIR(gm, *args, options=onnxmlir_options)
        return torch_onnxmlir_object(*args)

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


class TorchONNXMLIR:
    def __init__(self, gm: torch.fx.GraphModule, *args, **kwargs):
        global global_uncompilable_graphs
        # Input graph module.
        self.gm = gm
        self.gm.eval()

        # Indice of the actual example inputs if the graph's signature is changed.
        self.example_inputs_indices = None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Original graph module {self.gm}")

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
            self.example_inputs_indices = self.rewrite_gm_for_export(*args)
            self.cache_key = generate_hash_key(self.gm, kwargs["options"])
            self.gm.meta["om_hash"] = self.cache_key
            self.gm.meta["om_example_inputs_indices"] = self.example_inputs_indices

        # Cache the rewritten graph module.
        assert self.cache_key, "cache key does not exist"
        self.cached_session = global_session_cache.get(self.cache_key)
        if self.cached_session:
            self.example_inputs_indices = self.cached_session.example_inputs_indices
        elif self.cache_key in global_uncompilable_graphs:
            self.example_inputs_indices = self.gm.meta["om_example_inputs_indices"]
        assert self.example_inputs_indices, "example_inputs_indices is None"

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
        tensor_example_inputs = self.get_tensor_example_inputs(example_inputs)
        return self.forward(*tensor_example_inputs)

    def forward(self, *example_inputs):
        global global_uncompilable_graphs
        if self.cached_session is None:
            if self.cache_key in global_uncompilable_graphs:
                logger.info("Found the uncompilable model. Switch to the eager mode")
                return eager_forward_fn(self.gm)(*example_inputs)

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
            logger.info("Export and compile the model.")
            succeeded = self.export_gm_to_onnx(example_inputs)
            if not succeeded:
                logger.info("Failed to export the model. Switch to the eager mode.")
                global_uncompilable_graphs.add(self.cache_key)
                return eager_forward_fn(self.gm)(*example_inputs)

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

    def get_tensor_example_inputs(self, example_inputs):
        tensor_inputs = []
        for i in self.example_inputs_indices:
            x = example_inputs[i]
            if isinstance(x, int):
                tensor_inputs.append(torch.tensor(x, dtype=torch.int64).reshape((1,)))
            elif isinstance(x, torch.Tensor):
                tensor_inputs.append(x)
            else:
                raise ValueError("Unsupported input type. Consider to support it")
        return tuple(tensor_inputs)

    def build_dynamic_shapes_for_export(self) -> ([str], dict[str, dict[int, str]]):
        """
        This computes a dictionary of dynamic shapes to be used in torch.export.
        """
        dim_tables = {}
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
                dynamic_shapes[input_name] = {}
                continue
            # Get dynamic dimensions from dynamic input tensors.
            dynamic_dims = {}
            for dim_idx, dim_size in enumerate(input_arg.shape):
                if isinstance(dim_size, torch.SymInt):
                    if not str(dim_size).isdigit():
                        dim_str = "dim" + str(dim_size)
                        if dim_str in dim_tables:
                            dim = dim_tables[dim_str]
                        else:
                            dim = Dim(dim_str)
                            dim_tables[dim_str] = dim
                        dynamic_dims[dim_idx] = dim
            dynamic_shapes[input_name] = dynamic_dims
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"dynamic_shapes: {dynamic_shapes}")
        return input_names, dynamic_shapes

    def rewrite_gm_for_export(self, *example_inputs):
        n_ph_0 = len([n for n in self.gm.graph.nodes if n.op == "placeholder"])

        # Freeze scalar constant arguments that are typically parameters, e.g.,
        # epsilon value, from the config file of the model and they are constants.
        constant_values = self.extract_scalar_constant_args(example_inputs)
        self.gm = fx_utils.freeze_scalar_constant_args(self.gm, constant_values)

        # Since onnx does not support scalar inputs, symbolic integer arguments
        # are converted to tensor arguments.
        self.gm = fx_utils.convert_symint_args_to_tensors(self.gm)
        # Rewrite ops related to symbolic integers, e.g. torch.sym_sum.
        self.gm = fx_utils.rewrite_torch_sym_sum(self.gm)

        # Make sure that previous transformations did not change the number of placeholders.
        n_ph_1 = len([n for n in self.gm.graph.nodes if n.op == "placeholder"])
        assert n_ph_0 == n_ph_1, "The number of placeholders was changed"

        # Remove unused placeholders in one shot to get a consitent example_inputs_indices.
        self.gm, example_inputs_indices = fx_utils.remove_unused_placeholders(self.gm)

        return example_inputs_indices

    def extract_scalar_constant_args(self, example_inputs: tuple):
        """
        Extract scalar float constant arguments that are typically parameters, e.g.,
        epsilon value, from the config file of the model and they are constants.
        """
        graph = self.gm.graph
        placeholder_nodes = [n for n in graph.nodes if n.op == "placeholder"]
        input_names = [n.name for n in placeholder_nodes]

        # Map input names to example values.
        name_to_value = dict(zip(input_names, example_inputs))

        # Detect scalar constants by this pattern: placeholder -> .item().
        name_to_use_nodes = {}
        for node in placeholder_nodes:
            input_arg = node.meta["example_value"]
            # Not a tensor.
            if not isinstance(input_arg, torch.Tensor):
                continue
            # Not a float.
            if input_arg.dtype not in [torch.float32, torch.float64]:
                continue
            # Not a scalar.
            if input_arg.ndim != 0:
                continue
            # Pattern: placeholder -> .item().
            uses = [n for n in graph.nodes if node in n.all_input_nodes]
            item_nodes = [n for n in uses if fx_utils._is_item_of_tensor(n)]
            if not item_nodes:
                continue
            item_node = item_nodes[0]
            item_uses = [n for n in graph.nodes if item_node in n.all_input_nodes]
            other_uses = [n for n in uses if n != item_node]
            if item_uses and not other_uses:
                name_to_use_nodes[node.name] = item_nodes

        # Build constant_values dict.
        constant_values = {
            name: (name_to_value[name], name_to_use_nodes[name])
            for name in name_to_use_nodes.keys()
            if name in name_to_value
        }

        return constant_values

    def export_gm_to_onnx(self, example_inputs):
        model_name = self.default_model_name + str(self.tag) + ".onnx"
        self.onnx_model = os.path.join(self.workdir.name, model_name)
        input_names, dynamic_shapes = self.build_dynamic_shapes_for_export()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Fx Graph for exporting to onnx: {self.gm}")

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
                report=False,
            )
            succeeded = True
        except torch.onnx.errors.UnsupportedOperatorError as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"ONNX export unsupported: {e}")
                logger.debug(f"Fx Graph: {self.gm}")
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"ONNX export failure: {e}")
                logger.debug(f"Fx Graph: {self.gm}")

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
                os.remove(f)


# Alternative interface to minic the usage of torch.compile
def compile(torch_model, *args, **kwargs):
    return TorchONNXMLIR(torch_model, *args, **kwargs)
