import numpy as np
import os
import sys
import tempfile
import json
import subprocess


def get_names_in_signature(signature):
    names = []
    # Load the input signature.
    signature_dict = json.loads(signature)
    for sig in signature_dict:
        names.append(sig["name"])
    return names


class InferenceSession:
    def __init__(self, model_path, **kwargs):
        self.debug = False
        self.session = None
        self.handleParameters(model_path, **kwargs)
        if self.session is not None:
            return self.session
        self.checkCompiler()
        self.Compile()
        self.session = self.getSession()

    def handleParameters(self, model_path, **kwargs):
        if "debug" in kwargs.keys():
            self.debug = kwargs["debug"]
        self.model_path = model_path
        if model_path.endswith(".mlir"):
            self.model_suffix = ".mlir"
        elif model_path.endswith(".onnx"):
            self.model_suffix = ".onnx"
        elif model_path.endswith(".so"):
            self.compiled_lib = os.path.abspath(model_path)
            self.session = self.getSession()
            return
        else:
            print("Invalid input model path. Must end with .onnx or .mlir or .onnxtext")
            exit(1)

        absolute_path = os.path.abspath(self.model_path)
        self.model_basename = os.path.basename(absolute_path)
        self.model_dirname = os.path.dirname(absolute_path)

        if "compile_options" in kwargs.keys():
            self.compile_options = kwargs["compile_options"]
        else:
            self.compile_options = ""

        if "compiler_container" in kwargs.keys():
            self.compiler_container = kwargs["compiler_container"]
            if "compiler_path" not in kwargs.keys():
                print(
                    "Please specify the path to your compiler when you are not using the default image"
                )
                exit(1)
        else:
            # Default image
            # The compiler command may have different path in different image

            self.compiler_container = "ghcr.io/onnxmlir/onnx-mlir-dev"
            self.compiler_path = "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir"

            # Other possible default
            """
            self.compiler_container = "ghcr.io/onnxmlir/onnx-mlir"
            self.compiler_path = "/usr/local/bin/bin/onnx-mlir"

            self.compiler_container = "onnxmlir/onnx-mlir-dev"
            self.compiler_path = "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir"
            """

        if "compiler_path" in kwargs.keys():
            self.compiler_path = kwargs["compiler_path"]

    def checkCompiler(self):
        if self.compiler_container == None:
            if not os.path.exists(self.compiler_path):
                print("the compiler path does not exist: ", self.compiler_path)
                exit(-1)
        else:
            import docker

            self.container_client = docker.from_env()
            try:
                msg = self.container_client.containers.run(
                    self.compiler_container, "test -e " + self.compiler_path
                )
            except Exception as e:
                print(
                    "the compiler path does not exist in container: ",
                    self.compiler_path,
                )
                exit(-1)

    def Compile(self):
        # Temporary directory for compilation
        self.output_tempdir = tempfile.TemporaryDirectory()
        self.output_dirname = self.output_tempdir.name

        if self.compiler_container is None:
            # Use the uniform variable for local compiler and docker image
            self.container_model_dirname = self.model_dirname
            self.container_output_dirname = self.output_dirname
        else:
            # Path to mount the model and output in container
            self.container_model_dirname = "/myinput"
            self.container_output_dirname = "/myoutput"

        # Construct compilation command
        command_str = self.compiler_path

        # Compiled library
        if self.compile_options != "":
            command_str += " " + self.compile_options
        command_str += " " + os.path.join(
            self.container_model_dirname, self.model_basename
        )

        # ToFix: should use temporary directory for compilation, and
        # use "-o" to put the compiled library in the temporary directory.
        self.compiled_model = os.path.join(
            self.output_dirname,
            self.model_basename.removesuffix(self.model_suffix) + ".so",
        )
        command_str += " -o " + os.path.join(
            self.container_output_dirname,
            self.model_basename.removesuffix(self.model_suffix),
        )

        # Logically, the model directory could be mounted as read only.
        # But wrong time error occurred with "r" mode
        if self.compiler_container is None:
            subprocess.run(command_str.split(" "))
        else:
            import docker

            msg = self.container_client.containers.run(
                self.compiler_container,
                command_str,
                volumes={
                    self.model_dirname: {
                        "bind": self.container_model_dirname,
                        "mode": "rw",
                    },
                    self.output_dirname: {
                        "bind": self.container_output_dirname,
                        "mode": "rw",
                    },
                },
            )

    def getSession(self):
        # When the script is used in package onnxmlir, the files to be imported
        # are within the package. Path in the pakcage should be used.
        # Otherwise, env variable ONNX_MLIR_HOME is used to for import path
        if __package__ == "onnxmlir":
            try:
                from .PyRuntime import OMExecutionSession
            except ImportError:
                raise ImportError(" Error in importing PyRuntime for onnxmlir package")

        else:
            if not os.environ.get("ONNX_MLIR_HOME", None):
                raise RuntimeError(
                    "Environment variable ONNX_MLIR_HOME is not set, please set it to the path to "
                    "the HOME directory for onnx-mlir. The HOME directory for onnx-mlir refers to "
                    "the parent folder containing the bin, lib, etc sub-folders in which ONNX-MLIR "
                    "execu    tables and libraries can be found, typically `onnx-mlir/build/Debug`"
                )
            RUNTIME_DIR = os.path.join(os.environ["ONNX_MLIR_HOME"], "lib")
            sys.path.append(RUNTIME_DIR)
            try:
                from PyRuntime import OMExecutionSession
            except ImportError:
                raise ImportError(
                    "Looks like you did not build the PyRuntime target, build it by running `make PyRuntime`."
                    "You may need to set ONNX_MLIR_HOME to `onnx-mlir/build/Debug` since `make PyRuntime` outputs to `build/Debug` by default"
                )

        return OMExecutionSession(self.compiled_model, "NONE")

    def run(self, outputname, input_feed, **kwargs):
        inputs = []
        input_signature = self.session.input_signature()
        input_names = get_names_in_signature(input_signature)
        if input_feed:
            if isinstance(input_feed, dict):
                for name in input_names:
                    if name in input_feed:
                        inputs.append(input_feed[name])
                    else:
                        print("input name given: ", input_feed.keys())
                        print("input name expected by model: ", input_names)
                        print("do not match")
                        exit(1)
                # Since Python guarantees the order of values in a dictionary,
                # the name check could be ignored as follows:
                # inputs = list(input_feed.values())
            else:
                inputs = input_feed
        else:
            # Provide random value inputs
            print("error: input is not provided. ToFix: random input")
            exit(1)

        return self.session.run(inputs)
