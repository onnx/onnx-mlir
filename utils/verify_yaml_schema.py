#!/usr/bin/env python3
"""
Simple verification script for YAML schema files.
Checks structure without requiring ONNX/numpy dependencies.
"""

import yaml
import sys
import os


def verify_op_schema(op_dict, op_index):
    """Verify a single operation schema."""
    errors = []

    # Check required fields
    required_fields = ["name", "domain", "since_version", "support_level"]
    for field in required_fields:
        if field not in op_dict:
            errors.append(f"Op #{op_index} missing required field: {field}")

    op_name = op_dict.get("name", f"Op#{op_index}")

    # Check inputs
    if "inputs" in op_dict:
        for i, inp in enumerate(op_dict["inputs"]):
            if "name" not in inp:
                errors.append(f"{op_name}: input #{i} missing 'name'")
            if "type_str" not in inp:
                errors.append(
                    f"{op_name}: input '{inp.get('name', i)}' missing 'type_str'"
                )

    # Check outputs
    if "outputs" in op_dict:
        for i, out in enumerate(op_dict["outputs"]):
            if "name" not in out:
                errors.append(f"{op_name}: output #{i} missing 'name'")
            if "type_str" not in out:
                errors.append(
                    f"{op_name}: output '{out.get('name', i)}' missing 'type_str'"
                )

    # Check attributes
    if "attributes" in op_dict:
        for i, attr in enumerate(op_dict["attributes"]):
            if "name" not in attr:
                errors.append(f"{op_name}: attribute #{i} missing 'name'")
            if "type" not in attr:
                errors.append(
                    f"{op_name}: attribute '{attr.get('name', i)}' missing 'type'"
                )

    # Check type constraints
    if "type_constraints" in op_dict:
        for i, tc in enumerate(op_dict["type_constraints"]):
            if "type_param" not in tc:
                errors.append(f"{op_name}: type constraint #{i} missing 'type_param'")
            if "allowed_types" not in tc:
                errors.append(
                    f"{op_name}: type constraint #{i} missing 'allowed_types'"
                )

    # Check min/max
    for field in ["min_input", "max_input", "min_output", "max_output"]:
        if field in op_dict:
            if not isinstance(op_dict[field], int):
                errors.append(f"{op_name}: {field} must be an integer")

    return errors


def verify_yaml_file(yaml_path):
    """Verify a YAML schema file."""
    print(f"Verifying: {yaml_path}")
    print("=" * 60)

    if not os.path.exists(yaml_path):
        print(f"ERROR: File not found: {yaml_path}")
        return False

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"ERROR: YAML parsing failed: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Failed to read file: {e}")
        return False

    # Handle both list and single op
    if isinstance(data, list):
        ops_list = data
    else:
        ops_list = [data]

    print(f"Found {len(ops_list)} operation(s)\n")

    all_errors = []
    op_names = []

    for i, op_dict in enumerate(ops_list):
        op_name = op_dict.get("name", f"Op#{i}")
        op_names.append(op_name)

        print(f"  {i+1}. {op_name}")
        print(f"     Domain: {op_dict.get('domain', 'N/A')}")
        print(f"     Version: {op_dict.get('since_version', 'N/A')}")
        print(f"     Inputs: {len(op_dict.get('inputs', []))}")
        print(f"     Outputs: {len(op_dict.get('outputs', []))}")
        print(f"     Attributes: {len(op_dict.get('attributes', []))}")
        print(f"     Type Constraints: {len(op_dict.get('type_constraints', []))}")

        # Verify this op
        errors = verify_op_schema(op_dict, i)
        if errors:
            all_errors.extend(errors)
            print(f"     ✗ {len(errors)} error(s)")
            for error in errors:
                print(f"       - {error}")
        else:
            print(f"     ✓ Valid")
        print()

    print("=" * 60)
    if all_errors:
        print(f"✗ Verification FAILED with {len(all_errors)} error(s):")
        for error in all_errors:
            print(f"  - {error}")
        return False
    else:
        print(f"✓ Verification PASSED for {len(ops_list)} operation(s)")
        print(f"  Operations: {', '.join(op_names)}")
        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        yaml_path = os.path.join(os.path.dirname(__file__), "xfe_ops_schema.yaml")
        print(f"No file specified, using default: {yaml_path}\n")
    else:
        yaml_path = sys.argv[1]

    try:
        success = verify_yaml_file(yaml_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"UNEXPECTED ERROR: {e}")
        print(f"{'='*60}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
