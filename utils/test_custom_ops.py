#!/usr/bin/env python3
"""
Test script to verify custom ops YAML loading functionality.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from gen_onnx_mlir import load_custom_ops_from_yaml, CustomOpSchema


def test_load_xfe_ops():
    """Test loading XFE ops from YAML."""
    yaml_path = os.path.join(os.path.dirname(__file__), "xfe_ops_schema.yaml")

    if not os.path.exists(yaml_path):
        print(f"ERROR: {yaml_path} not found")
        return False

    print(f"Loading custom ops from: {yaml_path}")
    custom_ops = load_custom_ops_from_yaml(yaml_path)

    if not custom_ops:
        print("ERROR: No ops loaded")
        return False

    print(f"✓ Loaded {len(custom_ops)} custom operations")

    # Verify each op
    for op in custom_ops:
        print(f"\n  Op: {op.name}")
        print(f"    Domain: {op.domain}")
        print(f"    Version: {op.since_version}")
        print(f"    Inputs: {len(op.inputs)}")
        print(f"    Outputs: {len(op.outputs)}")
        print(f"    Attributes: {len(op.attributes)}")
        print(f"    Type Constraints: {len(op.type_constraints)}")

        # Verify required attributes
        assert op.name, f"Op missing name"
        assert op.domain, f"Op {op.name} missing domain"
        assert op.since_version >= 1, f"Op {op.name} has invalid version"

        # Verify inputs have required fields
        for inp in op.inputs:
            assert inp.name, f"Op {op.name} has input without name"
            assert hasattr(
                inp, "type_str"
            ), f"Op {op.name} input {inp.name} missing type_str"

        # Verify outputs have required fields
        for out in op.outputs:
            assert out.name, f"Op {op.name} has output without name"
            assert hasattr(
                out, "type_str"
            ), f"Op {op.name} output {out.name} missing type_str"

        print(f"    ✓ Validated")

    # Check for specific ops we know should be there
    op_names = [op.name for op in custom_ops]
    expected_ops = ["MatMulBias", "ConvChannelLast", "ConcatRuntime", "SliceRuntime"]

    for expected in expected_ops:
        if expected in op_names:
            print(f"\n✓ Found expected op: {expected}")
        else:
            print(f"\n✗ Missing expected op: {expected}")

    print(f"\n{'='*60}")
    print("✓ All tests passed!")
    print(f"{'='*60}")
    return True


if __name__ == "__main__":
    try:
        success = test_load_xfe_ops()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {e}")
        print(f"{'='*60}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
