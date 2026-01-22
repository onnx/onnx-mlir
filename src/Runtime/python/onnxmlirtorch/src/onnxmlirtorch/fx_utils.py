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
import torch
import torch.fx as fx
from typing import Optional


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
