# SPDX-License-Identifier: Apache-2.0

##################### onnx_utils.py *****#######################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################
#
# This file defines utility functions to rewrite an ONNX model.
#
################################################################################

import onnx


# ---------------------------------------
# Sanitize Identity nodes. The exporter sometimes produces
# Identity nodes whose input and output names are the same,
# which is invalid in ONNX.
# In this case, we rename the output.
# ---------------------------------------
def rename_identity_output(graph):
    new_nodes = []
    rename_map = {}
    counter = 0
    for node in graph.node:
        if node.op_type == "Identity":
            src = node.input[0]
            dst = node.output[0]
            if src == dst:
                # Same name: create new name.
                counter += 1
                rename_map[dst] = f"{dst}_id_{counter}"
            else:
                # Normal identity.
                rename_map[dst] = src
        else:
            new_nodes.append(node)

    # Nothing to do if all identity ops are well-defined.
    if counter == 0:
        return

    # Resolve chains.
    def resolve(name):
        visited = set()
        while name in rename_map and name not in visited:
            visited.add(name)
            name = rename_map[name]
        return name

    # Rewrite inputs.
    for node in new_nodes:
        for i in range(len(node.input)):
            node.input[i] = resolve(node.input[i])
    # Rewrite outputs.
    for out in graph.output:
        out.name = resolve(out.name)
    # Replace node list.
    graph.ClearField("node")
    graph.node.extend(new_nodes)


def sanitize_onnx(input_path, output_path):
    model = onnx.load(input_path)
    graph = model.graph

    # Sanitize Identity nodes. The exporter sometimes produces
    # Identity nodes whose input and output names are the same,
    # which is invalid in ONNX.
    rename_identity_output(graph)

    try:
        onnx.save_model(
            model,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
        )
    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Saving the sanitized ONNX model failed: {e}")
