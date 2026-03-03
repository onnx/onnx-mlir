import argparse
from .onnxmlirdockercompile import compile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m OMPyCompile",
        description="Command-line interface for package OMPyCompile.",
    )
    parser.add_argument(
        "model",
        type=str,
        help="model file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug output (default: False)",
    )
    parser.add_argument(
        "--compile_tag",
        type=str,
        default="NONE",
        help="Tag to pass to the compiler via --tag= (default: NONE)",
    )
    parser.add_argument(
        "--compile_options",
        type=str,
        default="",
        help="Additional options to pass to the onnx-mlir compiler (default: '')",
    )
    parser.add_argument(
        "--compiler_image_name",
        type=str,
        default="ghcr.io/onnxmlir/onnx-mlir-dev",
        help="Container image name to use for compilation",
    )
    parser.add_argument(
        "--container_engine",
        type=str,
        default=None,
        choices=["docker", "podman", None],
        help=(
            "Container engine to use: 'docker', 'podman', or None to auto-detect "
            "(default: None)"
        ),
    )
    parser.add_argument(
        "--compiler_path",
        type=str,
        default=None,
        help=(
            "Path to the onnx-mlir compiler binary. Required when using a custom "
            "compiler_image_name that is not in the built-in image dictionary."
        ),
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    kwargs = vars(args)
    model = kwargs.pop("model")
    compiled_model = compile(model, **kwargs)
    print(f"Compiled to {compiled_model}")

    return 0
