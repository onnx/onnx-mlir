#!/usr/bin/env python3

"""
Script to extract ONNX operation schemas and convert them to YAML format.
Usage: python gen_onnx_yaml.py --op MatMul --output matmul.yaml
       python gen_onnx_yaml.py --op MatMul --version 13
       python gen_onnx_yaml.py --all --output-dir ./onnx_ops/
"""
import argparse
import sys
import os
import yaml
from typing import Dict, Any, List, Optional

try:
    from onnx import defs
    from onnx.defs import OpSchema, ONNX_DOMAIN, ONNX_ML_DOMAIN
    import onnx
except ImportError:
    print("Error: ONNX package not found. Please install it with: pip install onnx")
    sys.exit(1)


def format_type_constraint(type_constraint):
    """Format a type constraint into a readable dict."""
    return {
        "type_param": type_constraint.type_param_str,
        "description": type_constraint.description,
        "allowed_types": list(type_constraint.allowed_type_strs),
    }


def format_formal_parameter(param, schema):
    """Format an input or output parameter."""
    result = {
        "name": param.name,
        "description": param.description,
        "type_str": param.type_str if param.type_str else None,
    }

    # Add option information
    if param.option == OpSchema.FormalParameterOption.Optional:
        result["optional"] = True
    elif param.option == OpSchema.FormalParameterOption.Variadic:
        result["variadic"] = True
        result["is_homogeneous"] = param.is_homogeneous

    return result


def format_attribute(attr_name, attr):
    """Format an attribute."""
    result = {
        "name": attr_name,
        "description": attr.description,
        "type": str(attr.type).split(".")[-1].lower(),
        "required": attr.required,
    }

    # Add default value if present
    if not attr.required and attr.default_value.name:
        from onnx import helper

        default_value = helper.get_attribute_value(attr.default_value)
        result["default_value"] = str(default_value)

    return result


def schema_to_dict(schema: OpSchema, include_docs: bool = True) -> Dict[str, Any]:
    """Convert an ONNX OpSchema to a dictionary."""

    result = {
        "name": schema.name,
        "domain": schema.domain if schema.domain else "ai.onnx",
        "since_version": schema.since_version,
        "support_level": str(schema.support_level).split(".")[-1],
    }

    # Add documentation if requested
    if include_docs and schema.doc:
        result["description"] = schema.doc.strip()

    # Add inputs
    if schema.inputs:
        result["inputs"] = [
            format_formal_parameter(inp, schema) for inp in schema.inputs
        ]

    # Add outputs
    if schema.outputs:
        result["outputs"] = [
            format_formal_parameter(out, schema) for out in schema.outputs
        ]

    # Add attributes
    if schema.attributes:
        result["attributes"] = [
            format_attribute(name, attr)
            for name, attr in sorted(schema.attributes.items())
        ]

    # Add type constraints
    if schema.type_constraints:
        result["type_constraints"] = [
            format_type_constraint(tc) for tc in schema.type_constraints
        ]

    # Add min/max input/output counts
    result["min_input"] = schema.min_input
    result["max_input"] = schema.max_input
    result["min_output"] = schema.min_output
    result["max_output"] = schema.max_output

    return result


def get_op_schema(
    op_name: str, domain: str = "", version: Optional[int] = None
) -> Optional[OpSchema]:
    """Get the schema for a specific operation."""

    # Get all schemas for this operation
    all_schemas = []
    for schema in defs.get_all_schemas_with_history():
        if schema.name == op_name and schema.domain == domain:
            all_schemas.append(schema)

    if not all_schemas:
        return None

    # Sort by version
    all_schemas.sort(key=lambda s: s.since_version)

    # Return specific version or latest
    if version is not None:
        for schema in all_schemas:
            if schema.since_version == version:
                return schema
        return None
    else:
        return all_schemas[-1]  # Return latest


def get_all_op_schemas(domain: str = "") -> List[OpSchema]:
    """Get all operation schemas for a domain."""
    schemas = {}

    for schema in defs.get_all_schemas_with_history():
        if schema.domain == domain:
            # Keep only the latest version of each op
            if (
                schema.name not in schemas
                or schema.since_version > schemas[schema.name].since_version
            ):
                schemas[schema.name] = schema

    return list(schemas.values())


def main():
    parser = argparse.ArgumentParser(
        description="Extract ONNX operation schemas to YAML format"
    )
    parser.add_argument(
        "--op", type=str, help="Operation name (e.g., MatMul, Conv, Add)"
    )
    parser.add_argument(
        "--version", type=int, help="Specific version/opset of the operation"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="",
        help='ONNX domain (default: "", ai.onnx.ml, etc.)',
    )
    parser.add_argument("--output", type=str, help="Output YAML file path")
    parser.add_argument("--all", action="store_true", help="Extract all operations")
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Output directory when using --all"
    )
    parser.add_argument(
        "--no-docs", action="store_true", help="Exclude documentation strings"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available operations"
    )

    args = parser.parse_args()

    # List operations
    if args.list:
        print(f"Available operations in ONNX {onnx.__version__}:")
        print("-" * 60)

        ops_by_domain = {}
        for schema in defs.get_all_schemas_with_history():
            domain = schema.domain if schema.domain else "ai.onnx"
            if domain not in ops_by_domain:
                ops_by_domain[domain] = {}
            if schema.name not in ops_by_domain[domain]:
                ops_by_domain[domain][schema.name] = []
            ops_by_domain[domain][schema.name].append(schema.since_version)

        for domain in sorted(ops_by_domain.keys()):
            print(f"\nDomain: {domain}")
            for op_name in sorted(ops_by_domain[domain].keys()):
                versions = sorted(ops_by_domain[domain][op_name])
                print(f"  {op_name}: versions {versions}")

        return

    include_docs = not args.no_docs

    # Extract all operations
    if args.all:
        schemas = get_all_op_schemas(args.domain)

        if not schemas:
            print(f"No operations found in domain '{args.domain}'")
            return

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        print(f"Extracting {len(schemas)} operations to {args.output_dir}/")
        for schema in schemas:
            op_dict = schema_to_dict(schema, include_docs)

            filename = f"{schema.name.lower()}.yaml"
            filepath = os.path.join(args.output_dir, filename)

            with open(filepath, "w") as f:
                yaml.dump(op_dict, f, default_flow_style=False, sort_keys=False)

            print(f"  Created: {filename}")

        print(f"\nSuccessfully extracted {len(schemas)} operations")
        return

    # Extract single operation
    if not args.op:
        parser.print_help()
        return

    schema = get_op_schema(args.op, args.domain, args.version)

    if not schema:
        if args.version:
            print(
                f"Error: Operation '{args.op}' version {args.version} not found in domain '{args.domain}'"
            )
        else:
            print(f"Error: Operation '{args.op}' not found in domain '{args.domain}'")
        sys.exit(1)

    # Convert to dict
    op_dict = schema_to_dict(schema, include_docs)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            yaml.dump(op_dict, f, default_flow_style=False, sort_keys=False)
        print(f"Schema written to: {args.output}")
    else:
        # Print to stdout
        yaml.dump(op_dict, sys.stdout, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
