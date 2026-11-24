#!/usr/bin/env python3
"""Replace SplitToSequence + SequenceAt patterns with Split in an ONNX model.

The pattern:
    seq = SplitToSequence(input, split_size, axis=A, keepdims=1)
    out0 = SequenceAt(seq, 0)
    out1 = SequenceAt(seq, 1)
    ...
becomes:
    out0, out1, ... = Split(input, split_size_repeated, axis=A)

Usage:
    python remove_sequence_ops.py input.onnx [output.onnx]
"""

import argparse
import sys

import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np


def remove_sequence_ops(path, out_path):
    m = onnx.load(path)
    g = m.graph

    # Build maps: output_name -> node, constant_name -> value
    prod = {o: n for n in g.node for o in n.output}
    consts = {}
    for n in g.node:
        if n.op_type == "Constant" and len(n.output) == 1:
            for a in n.attribute:
                if a.name == "value" and a.t.ByteSize() > 0:
                    consts[n.output[0]] = numpy_helper.to_array(a.t)
                elif a.name == "value_int":
                    consts[n.output[0]] = int(a.i)
                elif a.name == "value_ints":
                    consts[n.output[0]] = list(a.ints)
    for init in g.initializer:
        consts[init.name] = numpy_helper.to_array(init)

    # Find SplitToSequence nodes and their SequenceAt consumers
    splits_to_remove = {}  # SplitToSequence output name -> info
    for n in g.node:
        if n.op_type != "SplitToSequence":
            continue
        seq_out = n.output[0]
        axis = 0
        keepdims = 1
        for a in n.attribute:
            if a.name == "axis":
                axis = a.i
            elif a.name == "keepdims":
                keepdims = a.i

        # Gather all SequenceAt consumers of this SplitToSequence
        consumers = {}
        for n2 in g.node:
            if n2.op_type == "SequenceAt" and n2.input[0] == seq_out:
                pos_name = n2.input[1]
                if pos_name in consts:
                    pos = (
                        int(consts[pos_name])
                        if isinstance(consts[pos_name], (int, np.integer))
                        else int(consts[pos_name].item())
                    )
                    consumers[pos] = n2.output[0]
                else:
                    print(
                        f"WARNING: SequenceAt position {pos_name} is not a constant, skipping"
                    )
                    break
        else:
            if consumers:
                splits_to_remove[seq_out] = {
                    "node": n,
                    "axis": axis,
                    "keepdims": keepdims,
                    "consumers": consumers,
                }

    if not splits_to_remove:
        print("No SplitToSequence + SequenceAt patterns found")
        return

    # Build replacement nodes
    new_nodes = []
    nodes_to_remove = set()
    for seq_out, info in splits_to_remove.items():
        sn = info["node"]
        axis = info["axis"]
        consumers = info["consumers"]
        n_splits = max(consumers.keys()) + 1

        # Build Split output names: use the SequenceAt output name for consumed
        # positions, generate a dummy name for unconsumed ones
        split_outputs = []
        for i in range(n_splits):
            if i in consumers:
                split_outputs.append(consumers[i])
            else:
                split_outputs.append(f"{seq_out}__unused_{i}")

        # SplitToSequence accepts a scalar split_size, but Split needs a 1-D
        # tensor [size, size, ...] with one entry per output.  When the second
        # input is a scalar constant, replace it with the repeated vector.
        split_input_name = sn.input[0]
        if len(sn.input) > 1:
            size_name = sn.input[1]
            if size_name in consts:
                val = consts[size_name]
                scalar = (
                    int(val) if isinstance(val, (int, np.integer)) else int(val.item())
                )
                vec_name = f"{seq_out}__split_sizes"
                vec = numpy_helper.from_array(
                    np.array([scalar] * n_splits, dtype=np.int64), name=vec_name
                )
                g.initializer.append(vec)
                split_inputs = [split_input_name, vec_name]
            else:
                split_inputs = list(sn.input)
        else:
            split_inputs = [split_input_name]

        split_node = helper.make_node(
            "Split",
            inputs=split_inputs,
            outputs=split_outputs,
            name=f"{sn.name}_to_split",
            axis=axis,
        )
        new_nodes.append(split_node)

        # Mark SplitToSequence and its SequenceAt consumers for removal
        nodes_to_remove.add(id(sn))
        for n2 in g.node:
            if n2.op_type == "SequenceAt" and n2.input[0] == seq_out:
                nodes_to_remove.add(id(n2))

        print(
            f"Replaced SplitToSequence '{sn.name}' (axis={axis}) + "
            f"{len(consumers)} SequenceAt -> Split with {n_splits} outputs"
        )

    # Rebuild node list
    kept = [n for n in g.node if id(n) not in nodes_to_remove]
    # Insert new Split nodes at the position of the original SplitToSequence
    result = []
    new_iter = iter(new_nodes)
    next_new = next(new_iter, None)
    for n in kept:
        result.append(n)
        # After inserting a node that produces an input to a Split, check if
        # a Split should follow. We insert all new nodes right after removing
        # the old ones, so just append them at the position of the first kept
        # node that uses a Split output.

    # Simpler approach: insert new nodes where the SplitToSequence was
    result = []
    new_by_orig = {}
    for seq_out, info in splits_to_remove.items():
        for nn in new_nodes:
            if nn.name == f"{info['node'].name}_to_split":
                new_by_orig[id(info["node"])] = nn
    inserted = set()
    for n in g.node:
        if id(n) in nodes_to_remove:
            if id(n) in new_by_orig and id(n) not in inserted:
                result.append(new_by_orig[id(n)])
                inserted.add(id(n))
            continue
        result.append(n)

    del g.node[:]
    g.node.extend(result)

    # Clean up: remove sequence type info that may confuse validators
    used_outputs = {o for n in g.node for o in n.output}
    for vi in list(g.value_info):
        if vi.name not in used_outputs:
            continue
        if vi.type.HasField("sequence_type"):
            g.value_info.remove(vi)

    onnx.save(m, out_path)
    print(f"Saved to {out_path}")
    onnx.checker.check_model(out_path, full_check=True)
    print("Model check passed")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input", help="input ONNX model")
    p.add_argument(
        "output", nargs="?", help="output ONNX model (default: overwrite input)"
    )
    args = p.parse_args()
    out = args.output or args.input
    remove_sequence_ops(args.input, out)


if __name__ == "__main__":
    main()
