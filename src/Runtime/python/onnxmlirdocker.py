#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

##################### onnxmlirdocker.py ########################################
#
# Copyright 2019-2025 The IBM Research Authors.
#
################################################################################
#
# This script define a class to compile an onnx model with local compiler,
# or with a container image.
################################################################################

import numpy as np
import os
import sys
import tempfile
import json
import subprocess


class config:
    image_path_dictionary = {
        "ghcr.io/onnxmlir/onnx-mlir": "/usr/local/bin/bin/onnx-mlir",
        "ghcr.io/onnxmlir/onnx-mlir-dev": "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
        "onnxmlir/onnx-mlir-dev": "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
        # The entry point of zDLC image is the compiler
        "icr.io/ibmz/zdlc:5.0.0": "",
    }

    default_compiler_image_name = "ghcr.io/onnxmlir/onnx-mlir-dev"


def get_names_in_signature(signature):
    names = []
    # Load the input signature.
    signature_dict = json.loads(signature)
    for sig in signature_dict:
        names.append(sig["name"])
    return names


# Return the compiler path of an image based on the image_path_dictionary
def find_compiler_path(image_name):
    dict = config.image_path_dictionary
    if image_name in dict:
        return dict[image_name]
    else:
        return None


# Class to wrap the OMExecutionSession class with the compilation with
# container or local compiler.
# Logically, InferenceSession can be a subclass of OMExecutionSession.
# However, the OMExecutionSession is imported in a member function in current
# implementation. ToFix later.
class InferenceSession:
    def __init__(self, model_path, **kwargs):
        self.debug = False
        self.session = None
        self.output_dir = tempfile.TemporaryDirectory()
        self.handleParameters(model_path, **kwargs)
        if self.session is not None:
            return
        self.checkCompiler()
        self.Compile()
        self.session = self.getSession()

    def __del__(self):
        del self.session

    def handleParameters(self, model_path, **kwargs):
        if "debug" in kwargs.keys():
            self.debug = kwargs["debug"]
        if "compile_tag" in kwargs.keys():
            self.compile_tag = kwargs["compile_tag"]
        else:
            self.compile_tag = "NONE"

        self.model_path = model_path
        if model_path.endswith(".mlir"):
            self.model_suffix = ".mlir"
        elif model_path.endswith(".onnx"):
            self.model_suffix = ".onnx"
        elif model_path.endswith(".so"):
            self.compiled_model = os.path.abspath(model_path)
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

        # Parse the compile_options to handle the -o option.
        # If user does not specify the output, the compilation will be done in
        # the temporary directory.
        options_list = self.compile_options.split()

        # Logically, container_output_dirname should be used for -o
        if "-o" in options_list:
            # Convert the output to absolute path so that the compilation
            # can be done with compiler image.
            self.compiled_model = os.path.abspath(
                options_list[options_list.index("-o") + 1]
            )
            options_list[options_list.index("-o") + 1] = self.compiled_model

            self.compile_options = " ".join(options_list)
            if not self.compiled_model.endswith(".so"):
                self.compiled_model += ".so"
            self.output_dirname = os.path.dirname(self.compiled_model)
        else:
            self.output_dirname = self.output_dir.name
            self.compiled_model = os.path.join(
                self.output_dirname, self.model_basename.removesuffix(self.model_suffix)
            )
            self.compile_options += f" -o {self.compiled_model}"
            # Compile will automatically append .so to the -o target
            # Session need the suffix .so
            self.compiled_model += ".so"

        if "compiler_image_name" in kwargs.keys():
            self.compiler_image_name = kwargs["compiler_image_name"]
            if (
                self.compiler_image_name == "local"
                or self.compiler_image_name == "None"
            ):
                self.compiler_image_name = None
            self.compiler_path = find_compiler_path(self.compiler_image_name)
            if self.compiler_path is None and "compiler_path" not in kwargs.keys():
                print(
                    "Please specify the path to your compiler when you are not using the default image"
                )
                exit(1)
        else:
            # Default image
            self.compiler_image_name = config.default_compiler_image_name
            self.compiler_path = find_compiler_path(self.compiler_image_name)

        if "container_engine" in kwargs.keys():
            self.container_engine = kwargs["container_engine"]
            if (
                self.container_engine != "docker"
                and self.container_engine != "podman"
                and self.container_engine != None
            ):
                print(
                    "Container_engine has to be either 'docker' or 'podman', or None to let system choose the availabe one."
                )
                exit(1)
        else:
            self.container_engine = None

        if "compiler_path" in kwargs.keys():
            self.compiler_path = kwargs["compiler_path"]

    def checkCompiler(self):
        if self.compiler_image_name == None:
            if not os.path.exists(self.compiler_path):
                print("the compiler path does not exist: ", self.compiler_path)
                exit(-1)
        else:
            # Import container tool, either docker or podman package
            if self.container_engine is None:
                try:
                    import docker as ce
                except ImportError:
                    try:
                        import podman as ce
                    except ImportError:
                        raise ImportError(
                            "Failure to load docker or podman package. To install docker, you can either do 'pip install docker', or install onnxmlir with `pip install -e path/onnxmlir[docker]`. Similar for podman"
                        )
            elif self.container_engine == "docker":
                try:
                    import docker as ce
                except ImportError:
                    raise ImportError(
                        "Failure to load docker package. you can either do 'pip install docker', or install onnxmlir with `pip install -e path/onnxmlir[docker]`"
                    )
            else:
                try:
                    import podman as ce
                except ImportError:
                    raise ImportError(
                        "Failure to load podman package. you can either do 'pip install podman', or install onnxmlir with `pip install -e path/onnxmlir[podman]`"
                    )
            # The docker and podman package has the same interface
            # Get container client using env setting.
            self.container_client = ce.from_env()

            # Pull the image if not already available
            try:
                image = self.container_client.images.get(self.compiler_image_name)
            except ce.errors.ImageNotFound:
                image = self.container_client.images.pull(self.compiler_image_name)

            # Chek whether the specified compiler exists or not
            if os.path.exists(self.compiler_path):
                self.mount_compiler = True
            else:
                self.mount_compiler = False
                try:
                    msg = self.container_client.containers.run(
                        self.compiler_image_name, self.compiler_path + " --version"
                    )
                    # Could check the valid version of compiler
                except Exception as e:
                    print(
                        "the compiler path does not exist in container: ",
                        self.compiler_path,
                    )
                    exit(-1)

    def Compile(self):
        # Logically use different variable for path in current env and
        # container env. They could be different.
        # However, if the caller is already in a container, the path
        # on the host has to be used. Therefore, it is easier to always use
        # the same path to mount on container.
        self.container_model_dirname = self.model_dirname
        self.container_output_dirname = self.output_dirname

        # Construct compilation command
        command_str = self.compiler_path

        # Compiled library
        if self.compile_options != "":
            command_str += " " + self.compile_options

        command_str += " --tag=" + self.compile_tag

        command_str += " " + os.path.join(
            self.container_model_dirname, self.model_basename
        )

        # print(command_str)

        # Logically, the model directory could be mounted as read only.
        # But wrong time error occurred with "r" mode
        if self.compiler_image_name is None:
            subprocess.run(command_str.split(" "))
            self.container = None
        else:
            # ToFix: try detach=True?
            try:
                msg = self.container_client.containers.run(
                    self.compiler_image_name,
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
                        # Mount the compiler if needed.
                        # ToFix: compiler built with container may fail, possibly because
                        # the directory is mounted differently when the compiler is built
                        # with container image.
                        # A good practice is to always mount with the original path
                        **(
                            {
                                os.path.dirname(self.compiler_path): {
                                    "bind": os.path.dirname(self.compiler_path),
                                    "mode": "rw",
                                }
                            }
                            if self.mount_compiler
                            else {}
                        ),
                    },
                )
            except Exception as e:
                print(
                    f"Failed in compilation with docker image: {self.compiler_image_name}"
                )
                print(f"Here is the compiling command: {command_str}")
                print(
                    "Check whether the flags are correct, the input file exists, and the output file is writable"
                )
                exit(-1)

    def getSession(self):
        # When the script is used in package onnxmlir, the files to be imported
        # are within the package. Path in the package should be used.
        # Otherwise, env variable ONNX_MLIR_HOME is used to for import path
        if __package__ == "onnxmlir" or __package__ == "onnxmlirtorch":
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
                    "Looks like you did not build the PyRuntimeC target, build it by running `make PyRuntimeC`."
                    "You may need to set ONNX_MLIR_HOME to `onnx-mlir/build/Debug` since `make PyRuntimeC` outputs to `build/Debug` by default"
                )

        return OMExecutionSession(self.compiled_model, self.compile_tag)

    # wrapper for onnxruntime interface
    def run_ort(self, outputname, input_feed, **kwargs):
        return self.run(input_feed, **kwargs)

    def run(self, input_feed, **kwargs):
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

        # Let onnx-mlir know where to find the constants file.
        os.environ["OM_CONSTANT_PATH"] = os.path.dirname(self.compiled_model)
        return self.session.run(inputs)

    def input_signature(self):
        return self.session.input_signature()

    def output_signature(self):
        return self.session.output_signature()
