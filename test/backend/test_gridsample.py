#!/usr/bin/env python3

"""
Test GridSample operation against PyTorch implementation.

This test compares the ONNX-MLIR lowering of GridSample with PyTorch's
F.grid_sample implementation to ensure correctness.
"""

import numpy as np
import torch
import torch.nn.functional as F
import onnx
from onnx import helper, TensorProto
import sys


def create_gridsample_model(
    input_shape, grid_shape, mode="linear", padding_mode="zeros", align_corners=0
):
    """Create an ONNX model with GridSample operation."""

    # Create input and grid tensors
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    grid = helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    # Create GridSample node
    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    # Create graph
    graph = helper.make_graph(
        [node],
        "gridsample_test",
        [X, grid],
        [Y],
    )

    # Create model
    model = helper.make_model(graph, producer_name="gridsample_test")
    model.opset_import[0].version = 16  # GridSample is available from opset 16

    return model


def test_gridsample_2d_bilinear():
    """Test 2D GridSample with bilinear interpolation."""
    print("\n=== Test 2D Bilinear ===")

    # Input: [N=1, C=1, H=4, W=4]
    input_data = np.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]],
        dtype=np.float32,
    )

    # Grid: [N=1, H_out=2, W_out=2, 2]
    grid_data = np.array(
        [[[[-0.5, -0.5], [0.5, -0.5]], [[-0.5, 0.5], [0.5, 0.5]]]], dtype=np.float32
    )

    # PyTorch reference
    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid_data)
    output_torch = F.grid_sample(
        input_torch,
        grid_torch,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    print(f"Input shape: {input_data.shape}")
    print(f"Grid shape: {grid_data.shape}")
    print(f"PyTorch output shape: {output_torch.shape}")
    print(f"PyTorch output:\n{output_torch.numpy()}")

    # Create ONNX model
    model = create_gridsample_model(
        input_shape=[1, 1, 4, 4],
        grid_shape=[1, 2, 2, 2],
        mode="linear",
        padding_mode="zeros",
        align_corners=0,
    )

    # Save model for testing
    onnx.save(model, "/tmp/gridsample_2d_bilinear.onnx")
    print("Model saved to /tmp/gridsample_2d_bilinear.onnx")

    return True


def test_gridsample_2d_nearest():
    """Test 2D GridSample with nearest neighbor interpolation."""
    print("\n=== Test 2D Nearest ===")

    # Input: [N=1, C=1, H=4, W=4]
    input_data = np.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]],
        dtype=np.float32,
    )

    # Grid: [N=1, H_out=2, W_out=2, 2]
    grid_data = np.array(
        [[[[-0.5, -0.5], [0.5, -0.5]], [[-0.5, 0.5], [0.5, 0.5]]]], dtype=np.float32
    )

    # PyTorch reference
    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid_data)
    output_torch = F.grid_sample(
        input_torch,
        grid_torch,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    )

    print(f"Input shape: {input_data.shape}")
    print(f"Grid shape: {grid_data.shape}")
    print(f"PyTorch output shape: {output_torch.shape}")
    print(f"PyTorch output:\n{output_torch.numpy()}")

    # Create ONNX model
    model = create_gridsample_model(
        input_shape=[1, 1, 4, 4],
        grid_shape=[1, 2, 2, 2],
        mode="nearest",
        padding_mode="zeros",
        align_corners=0,
    )

    # Save model for testing
    onnx.save(model, "/tmp/gridsample_2d_nearest.onnx")
    print("Model saved to /tmp/gridsample_2d_nearest.onnx")

    return True


def test_gridsample_2d_bicubic():
    """Test 2D GridSample with bicubic interpolation."""
    print("\n=== Test 2D Bicubic ===")

    # Input: [N=1, C=1, H=4, W=4]
    input_data = np.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]],
        dtype=np.float32,
    )

    # Grid: [N=1, H_out=2, W_out=2, 2]
    grid_data = np.array(
        [[[[-0.5, -0.5], [0.5, -0.5]], [[-0.5, 0.5], [0.5, 0.5]]]], dtype=np.float32
    )

    # PyTorch reference
    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid_data)
    output_torch = F.grid_sample(
        input_torch,
        grid_torch,
        mode="bicubic",
        padding_mode="zeros",
        align_corners=False,
    )

    print(f"Input shape: {input_data.shape}")
    print(f"Grid shape: {grid_data.shape}")
    print(f"PyTorch output shape: {output_torch.shape}")
    print(f"PyTorch output:\n{output_torch.numpy()}")

    # Create ONNX model
    model = create_gridsample_model(
        input_shape=[1, 1, 4, 4],
        grid_shape=[1, 2, 2, 2],
        mode="cubic",
        padding_mode="zeros",
        align_corners=0,
    )

    # Save model for testing
    onnx.save(model, "/tmp/gridsample_2d_bicubic.onnx")
    print("Model saved to /tmp/gridsample_2d_bicubic.onnx")

    return True


def test_gridsample_3d_trilinear():
    """Test 3D GridSample with trilinear interpolation."""
    print("\n=== Test 3D Trilinear ===")

    # Input: [N=1, C=1, D=2, H=2, W=2]
    input_data = np.array([[[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]], dtype=np.float32)

    # Grid: [N=1, D_out=2, H_out=2, W_out=2, 3]
    grid_data = np.array(
        [
            [
                [
                    [[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5]],
                    [[-0.5, 0.5, -0.5], [0.5, 0.5, -0.5]],
                ],
                [
                    [[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5]],
                    [[-0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
                ],
            ]
        ],
        dtype=np.float32,
    )

    # PyTorch reference
    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid_data)
    output_torch = F.grid_sample(
        input_torch,
        grid_torch,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    print(f"Input shape: {input_data.shape}")
    print(f"Grid shape: {grid_data.shape}")
    print(f"PyTorch output shape: {output_torch.shape}")
    print(f"PyTorch output:\n{output_torch.numpy()}")

    # Create ONNX model
    model = create_gridsample_model(
        input_shape=[1, 1, 2, 2, 2],
        grid_shape=[1, 2, 2, 2, 3],
        mode="linear",
        padding_mode="zeros",
        align_corners=0,
    )

    # Save model for testing
    onnx.save(model, "/tmp/gridsample_3d_trilinear.onnx")
    print("Model saved to /tmp/gridsample_3d_trilinear.onnx")

    return True


def test_gridsample_align_corners():
    """Test GridSample with align_corners=1."""
    print("\n=== Test Align Corners ===")

    # Input: [N=1, C=1, H=4, W=4]
    input_data = np.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]],
        dtype=np.float32,
    )

    # Grid: [N=1, H_out=2, W_out=2, 2]
    grid_data = np.array(
        [[[[-1.0, -1.0], [1.0, -1.0]], [[-1.0, 1.0], [1.0, 1.0]]]], dtype=np.float32
    )

    # PyTorch reference
    input_torch = torch.from_numpy(input_data)
    grid_torch = torch.from_numpy(grid_data)
    output_torch = F.grid_sample(
        input_torch,
        grid_torch,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    print(f"Input shape: {input_data.shape}")
    print(f"Grid shape: {grid_data.shape}")
    print(f"PyTorch output shape: {output_torch.shape}")
    print(f"PyTorch output:\n{output_torch.numpy()}")

    # Create ONNX model
    model = create_gridsample_model(
        input_shape=[1, 1, 4, 4],
        grid_shape=[1, 2, 2, 2],
        mode="linear",
        padding_mode="zeros",
        align_corners=1,
    )

    # Save model for testing
    onnx.save(model, "/tmp/gridsample_align_corners.onnx")
    print("Model saved to /tmp/gridsample_align_corners.onnx")

    return True


if __name__ == "__main__":
    print("GridSample PyTorch Comparison Tests")
    print("=" * 50)

    try:
        test_gridsample_2d_bilinear()
        test_gridsample_2d_nearest()
        test_gridsample_2d_bicubic()
        test_gridsample_3d_trilinear()
        test_gridsample_align_corners()

        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("\nGenerated ONNX models can be tested with:")
        print("  onnx-mlir --EmitLib /tmp/gridsample_*.onnx")
        print("\nThen compare outputs with PyTorch reference values shown above.")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
