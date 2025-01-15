#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

########################## onnx-mlir.py ########################################
#
# Copyright 2022 The IBM Research Authors.
#
################################################################################
#
# This file scans for certain patterns (listed below) and generate an md table,
# which list the operations supported, and optionally the unsupported operations.
# Among the options, we can also list the TODOs in the table.
# Invoke with the `-h` argument to list options.
#
# Limitation: currently handle at most one OP/LIMIT/TODO line per operation.
# Script currently invoked by the `onnx_mlir_supported_ops` make target.
#
################################################################################

# When running onnx-mlir inside a docker container, the directory
# containing the input ONNX model file must be mounted into the
# container for onnx-mlir to read the input and generate the output.
#
# This convenient script will do that automatically to make it
# as if you are running onnx-mlir directly on the host.

import os
import re
import shutil
import stat
import subprocess
import sys

DOCKER_SOCKET = "/var/run/docker.sock"
ONNX_MLIR_IMAGE = "ghcr.io/onnxmlir/onnx-mlir"
KNOWN_INPUT_TYPE = (".onnx", ".json", ".mlir")

mount_dirs = []
mount_args = []
onnx_mlir_args = []


# mount host path into container
def mount_path(path):
    global mount_dirs, mount_args, onnx_mlir_args

    p = os.path.abspath(path)
    d = os.path.dirname(p)
    f = os.path.basename(p)

    # Haven't seen this directory before
    if not d in mount_dirs:
        mount_dirs += [d]
        mount_args += ["-v", d + ":" + d]
        onnx_mlir_args += [p]
    else:
        onnx_mlir_args += [path]


def main():
    # Make sure docker client is installed
    if not shutil.which("docker"):
        print("docker client not found")
        sys.exit(1)

    # Make sure docker daemon is running
    if not stat.S_ISSOCK(os.stat(DOCKER_SOCKET).st_mode):
        print("docker daemon not running")
        sys.exit(1)

    # Pull the latest onnxmlir/onnx-mlir image, if image
    # is already up-to-date, pull will do nothing.
    args = ["docker", "pull", ONNX_MLIR_IMAGE]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Only print messages related to pulling a layer or error.
    for l in proc.stdout:
        line = l.decode("utf-8")
        print(
            line if re.match("^([0-9a-f]{12})|Error", line) else "", end="", flush=True
        )
    proc.wait()
    if proc.returncode:
        print("docker pull failed")
        sys.exit(proc.returncode)

    # Prepare the arguments for docker run
    args = ["docker", "run", "--rm", "-ti"]

    # Go through the command line options and locate the known
    # file types. For each file located, construct a docker mount
    # option that mounts the host directory into the container.
    #
    # Also do the same for the output path specified by the -o
    # option.
    argv = sys.argv
    argc = len(sys.argv)

    global mount_dirs, mount_args, onnx_mlir_args

    verbose = False
    for i in range(1, argc):
        if argv[i].endswith(KNOWN_INPUT_TYPE):
            mount_path(argv[i])
        elif argv[i - 1] == "-o" and not argv[i].startswith("-"):
            mount_path(argv[i])
        elif argv[i] == "-v":
            verbose = True
            onnx_mlir_args += [argv[i]]
        else:
            onnx_mlir_args += [argv[i]]

    # Add effective uid and gid
    args += ["-u", str(os.geteuid()) + ":" + str(os.getegid())]

    # Add mount options
    args += mount_args

    # Add image name
    args += [ONNX_MLIR_IMAGE]

    # Pass in all the original arguments
    args += onnx_mlir_args

    if verbose:
        print(args)

    # Run onnx-mlir in the container
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        print(line.decode("utf-8"), end="", flush=True)
    proc.wait()
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
