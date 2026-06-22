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
# ---------------------------------------
def remove_identity_ops(graph):
    new_nodes = []
    rename_map = {}
    counter = 0
    for node in graph.node:
        if node.op_type == "Identity":
            src = node.input[0]
            dst = node.output[0]
            rename_map[dst] = src
            counter += 1
        else:
            new_nodes.append(node)

    # Nothing to do if there is no identity op.
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


# Sanitize an ONNX model and save it to disk.
def sanitize_onnx(input_path, output_path):
    model = onnx.load(input_path)
    graph = model.graph

    # Sanitize Identity nodes. The exporter sometimes produces
    # Identity nodes whose input and output names are the same,
    # which is invalid in ONNX.
    remove_identity_ops(graph)

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
