# SPDX-License-Identifier: Apache-2.0

##################### fx_utils.py *******#######################################
#
# Copyright 2025 The IBM Research Authors.
#
################################################################################
#
# This file defines utility functions to rewrite an FX graph.
#
################################################################################

import operator
import logging
from typing import Optional

import torch
import torch.fx as fx
from torch.fx.experimental.symbolic_shapes import (
    sym_eq,
)

logger = logging.getLogger(__name__)


def _get_dtype_from_meta(node: fx.Node) -> Optional[torch.dtype]:
    """
    Read dtype from FakeTensor meta attached to the node.
    Returns None if unavailable.
    """
    try:
        val = node.meta.get("tensor_meta", None)
        if val is not None and "dtype" in val:
            return val["dtype"]
    except Exception:
        pass
    return None


def _is_singleton_from_meta(node: fx.Node) -> bool:
    """
    Returns True if meta proves the tensor has exactly one element.
    Otherwise, False.
    """
    val = node.meta.get("tensor_meta", None)
    if val is None:
        return False

    # 0-d scalar tensor is trivially a singleton
    if "shape" in val:
        if len(val["shape"]) == 0:
            return True
        if len(val["shape"]) == 1 and val["shape"][0] == 1:
            return True

    return False


def _is_item_of_tensor(n: fx.Node) -> bool:
    return (
        n.op == "call_method"
        and n.target == "item"
        and len(n.args) == 1
        and isinstance(n.args[0], fx.Node)
    )


def freeze_scalar_constant_args(
    gm: fx.GraphModule, constant_values: dict
) -> fx.GraphModule:
    """
    Freeze scalar constant arguments that are typically parameters, e.g., epsilon value,
    from the config file of the model and they are constants.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"freeze_scalar_constant_args, constant_values: {constant_values}")

    if not constant_values:
        return gm

    graph = gm.graph
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
            gm.register_buffer(name, value)
        else:
            setattr(gm, name, value)

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
            if _is_item_of_tensor(use_node):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"freeze_scalar_constant_args, replace {use_node} by {value}"
                    )
                # Replace the .item() node with the scalar directly.
                scalar_value = (
                    value.item() if isinstance(value, torch.Tensor) else value
                )
                use_node.replace_all_uses_with(scalar_value)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"freeze_scalar_constant_args, {name}, {value} END")

    graph.lint()
    gm.recompile()
    return gm


def remove_unused_placeholders(gm: fx.GraphModule) -> (fx.GraphModule, list[int]):
    """
    Remove unused placeholders and return the indices of the remaining placeholders.
    """
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]

    unused_placeholder_indices = set()
    for i, n in enumerate(placeholders):
        if len(n.users) != 0:
            continue
        unused_placeholder_indices.add(i)
        gm.graph.erase_node(n)

    if unused_placeholder_indices:
        gm.graph.lint()
        gm.recompile()

    example_inputs_indices = [
        i for i in range(len(placeholders)) if i not in unused_placeholder_indices
    ]
    return gm, example_inputs_indices


def rewrite_torch_sym_sum(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Rewrite the exact pattern:

        item_s  = s_tensor.item()
        sym_sum = torch.sym_sum([1, item_s]); item_s = None

    into:

        one_i64 = torch.ones_like(s_tensor, dtype=torch.int64)
        out     = torch.add(s_tensor, one_i64)
        sym_sum = out.item()

    **Only** when:
      - dtype(s_tensor) is torch.int64, and
      - s_tensor is statically known to be a single-element tensor.

    If either condition cannot be proven from FX meta, the pattern is skipped.
    """
    g = gm.graph
    changed = False

    for node in g.nodes:
        # Find: call_function[target=torch.sym_sum] with a 2-element list/tuple [1, item_s] or [item_s, 1]
        if node.op == "call_function" and node.target is torch.sym_sum:
            # Match the pattern.
            if not node.args:
                continue
            seq = node.args[0]
            if not isinstance(seq, (tuple, list)) or len(seq) != 2:
                continue

            a, b = seq
            item_node: Optional[fx.Node] = None
            if a == 1 and isinstance(b, fx.Node) and _is_item_of_tensor(b):
                item_node = b
            elif b == 1 and isinstance(a, fx.Node) and _is_item_of_tensor(a):
                item_node = a
            else:
                continue

            # s_tensor is the receiver of .item()
            s_tensor = item_node.args[0]

            # dtype is int64.
            if _get_dtype_from_meta(s_tensor) is not torch.int64:
                continue

            # s_tensor is a scalar tensor.
            if not _is_singleton_from_meta(s_tensor):
                continue

            # Rewrite the pattern.
            with g.inserting_before(node):
                # one_i64 = ones_like(s_tensor, dtype=int64)
                one_i64 = g.call_function(
                    torch.ops.aten.ones_like.default,
                    args=(s_tensor,),
                    kwargs=dict(
                        dtype=torch.int64,
                        layout=None,
                        device=None,
                        pin_memory=None,
                        memory_format=None,
                    ),
                )
                # out = add(s_tensor, one_i64)
                out = g.call_function(
                    torch.ops.aten.add.Tensor, args=(s_tensor, one_i64), kwargs={}
                )
                # sym_sum = out.item()
                out_item = g.call_method("item", args=(out,), kwargs={})

            # Redirect uses and clean up old nodes.
            node.replace_all_uses_with(out_item)
            g.erase_node(node)
            if len(item_node.users) == 0:
                g.erase_node(item_node)

            changed = True

    if changed:
        g.lint()
        gm.recompile()
    return gm


def convert_symint_args_to_tensors(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Convert symbolic integer arguments into tensor arguments of type [1xint64].
    """
    # Important note: do not cast SymInt to int by int(SymInt)
    # since that concretizes symbolic dimensions in related Tensors.
    changed = False

    graph = gm.graph
    symint_placeholders = []
    tensor_placeholders = []

    # Collect SymInt placeholders.
    for node in graph.nodes:
        if node.op != "placeholder":
            continue
        if node.type in [torch.SymInt]:
            with graph.inserting_before(node):
                tensor_node = graph.placeholder(f"{node.name}_tensor")
                tensor_node.meta = node.meta
                tensor_node.meta["tensor_meta"] = {
                    "sizes": (1,),
                    "dtype": torch.int64,
                }
                tensor_node.type = torch.Tensor
                changed = True
            symint_placeholders.append((node, tensor_node))
        elif "example_value" in node.meta and isinstance(
            node.meta["example_value"], torch.Tensor
        ):
            tensor_placeholders.append(node)

    # Find which input the SymInt comes from, say, from which dim of which input.
    symint_carriers = {}
    for symint_node, _ in symint_placeholders:
        symint = symint_node.meta.get("example_value", None)
        if symint is None:
            continue
        found = False
        for node in tensor_placeholders:
            if found:
                break
            for i, d in enumerate(node.meta["example_value"].shape):
                if found:
                    break
                if isinstance(d, torch.SymInt) and sym_eq(d, symint):
                    symint_carriers[symint_node] = (node, i)
                    found = True

    # Remove the SymInt input and get the dimension size from the tensor node if possible.
    # Otherwise replace its uses with .item() calls.
    for symint_node, tensor_node in symint_placeholders:
        for user in list(symint_node.users):
            with graph.inserting_before(user):
                if symint_node in symint_carriers:
                    carrier = symint_carriers[symint_node][0]
                    dim = symint_carriers[symint_node][1]
                    item_node = graph.call_function(
                        torch.ops.aten.sym_size, args=(carrier, dim)
                    )
                else:
                    item_node = graph.call_method("item", args=(tensor_node,))
                user.replace_input_with(symint_node, item_node)
                changed = True
        graph.erase_node(symint_node)

    if changed:
        gm.graph.lint()
        gm.recompile()

    return gm
