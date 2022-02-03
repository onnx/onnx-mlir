<!--- SPDX-License-Identifier: Apache-2.0 -->
<p align="center"><img width="50%" src="docs/logo/onnx-mlir-1280x640.png" /></p>

# ONNX-MLIR

This project (https://onnx.ai/onnx-mlir/) provides compiler technology to transform a valid Open Neural Network Exchange (ONNX) graph into code that implements the graph with minimum runtime support.
It implements the [ONNX standard](https://github.com/onnx/onnx#readme) and is based on the underlying [LLVM/MLIR](https://mlir.llvm.org) compiler technology.

| System        | Build Status |
|---------------|--------------|
| s390x-Linux   | [![Build Status](https://www.onnxmlir.xyz/jenkins/buildStatus/icon?job=ONNX-MLIR-Pipeline-Docker-Build&build=last:%24%7Bparams.GITHUB_PR_NUMBER_PUSH=main%7D&subject=Jenkins%20CI)](https://www.onnxmlir.xyz/jenkins/job/ONNX-MLIR-Pipeline-Docker-Build/)             |
| ppc64le-Linux | [![Build Status](https://www.onnxmlir.xyz/jenkinp/buildStatus/icon?job=ONNX-MLIR-Pipeline-Docker-Build&build=last:%24%7Bparams.GITHUB_PR_NUMBER_PUSH=main%7D&subject=Jenkins%20CI)](https://www.onnxmlir.xyz/jenkinp/job/ONNX-MLIR-Pipeline-Docker-Build/)             |
| amd64-Linux   | [![Build Status](https://www.onnxmlir.xyz/jenkinx/buildStatus/icon?job=ONNX-MLIR-Pipeline-Docker-Build&build=last:%24%7Bparams.GITHUB_PR_NUMBER_PUSH=main%7D&subject=Jenkins%20CI)](https://www.onnxmlir.xyz/jenkinx/job/ONNX-MLIR-Pipeline-Docker-Build/)             |
| amd64-Windows | [![Build Status](https://dev.azure.com/onnx-pipelines/onnx/_apis/build/status/MLIR-Windows-CI?branchName=main)](https://dev.azure.com/onnx-pipelines/onnx/_build/latest?definitionId=9&branchName=main)             |
| amd64-macOS   | [![Build Status](https://github.com/onnx/onnx-mlir/actions/workflows/macos-amd64-build.yml/badge.svg)](https://github.com/onnx/onnx-mlir/actions/workflows/macos-amd64-build.yml)             |
|  | [![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5549/badge)](https://bestpractices.coreinfrastructure.org/projects/5549) |

This project contributes:
* an ONNX Dialect that can be integrated in other projects,
* a compiler interfaces that lower ONNX graphs into MLIR files/LLVM bytecodes/C & Java libraries,
* an `onnx-mlir` driver to perform these lowering,
* and a python/C/C++/Java runtime environment.

## Setting up ONNX-MLIR using Prebuilt Containers

The preferred approach to using and developing ONNX-MLIR is to use Docker Images and Containers, as getting the proper code dependences may be tricky on some systems. Our instructions on using ONNX-MLIR with Dockers are [here](docs/Docker.md).

If you intend to develop code, you should look at our [workflow](docs/Workflow.md) document which help you setup your Docker environment in a way that let you contribute code easily.

## Setting up ONNX-MLIR directly

ONNX-MLIR runs natively on Linux, OSX, and Windows.
Detailed instructions are provided below.

### Prerequisites

<!-- Keep list below in sync with docs/Prerequisite.md. -->
```
gcc >= 6.4
libprotoc >= 3.11.0
cmake >= 3.15.4
ninja >= 1.10.2
```

Help to update the prerequisites is found [here](docs/Prerequisite.md).

At any point in time, ONNX-MLIR depends on a specific commit of the LLVM project that has been shown to work with the project. 
Periodically the maintainers need to move to a more recent LLVM level. 
Among other things, this requires to update the commit string in (utils/clone-mlir.sh). 
When updating ONNX-MLIR, it is good practice to check that the commit string of the MLIR/LLVM is the same as the one listed in that file.

### Build on Linux or OSX

Directions to install MLIR and ONNX-MLIR are provided [here](docs/BuildOnLinuxOSX.md).

### Build on Windows

Directions to install Protobuf, MLIR, and ONNX-MLIR are provided [here](docs/BuildOnWindows.md).

### Testing build and summary of custom environment variables

After installation, an `onnx-mlir` executable should appear in the `build/Debug/bin` or `build/Release/bin` directory.

There are several cmake targets that are used to verify the validity of the `onnx-mlir` compiler, which are listed [here](docs/TestingHighLevel.md).

## Using ONNX-MLIR

The usage of `onnx-mlir` is as such:

```
OVERVIEW: ONNX-MLIR modular optimizer driver

USAGE: onnx-mlir [options] <input file>

OPTIONS:

Generic Options:

  --help        - Display available options (--help-hidden for more)
  --help-list   - Display list of available options (--help-list-hidden for more)
  --version     - Display the version of this program

ONNX-MLIR Options:
These are frontend options.

  Choose target to emit:
      --EmitONNXBasic - Ingest ONNX and emit the basic ONNX operations without inferred shapes.
      --EmitONNXIR    - Ingest ONNX and emit corresponding ONNX dialect.
      --EmitMLIR      - Lower the input to MLIR built-in transformation dialect.
      --EmitLLVMIR    - Lower the input to LLVM IR (LLVM MLIR dialect).
      --EmitObj       - Compile the input to an object file.      
      --EmitLib       - Compile and link the input into a shared library (default).
      --EmitJNI       - Compile the input to a jar file.

  Optimization levels:
      --O0           - Optimization level 0 (default).
      --O1           - Optimization level 1.
      --O2           - Optimization level 2.
      --O3           - Optimization level 3.
```

The full list of options is given by the `--help` option. Note that just as most compilers, the default optimization level is `-O0`. 
We recommend using `-O3` for most applications.

### Simple Example

For example, use the following command to lower an ONNX model (e.g., add.onnx) to ONNX dialect:
```shell
./onnx-mlir --EmitONNXIR add.onnx
```
The output should look like:
```mlir
module {
  func @main_graph(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
    return %0 : tensor<10x10x10xf32>
  }
}
```

An example based on the add operation is found [here](docs/doc_example), which build an ONNX model using a python script, and then provide a main program to load the model's value, compute, and print the models output.

### End to End Example

An end to end example is provided [here](docs/mnist_example/README.md), which train, compile, and execute a simple MNIST example using both the C++ or Python interface.

## Interacting via Slack and GitHub.

We have a slack channel established under the Linux Foundation AI and Data Workspace, named `#onnx-mlir-discussion`.
This channel can be used for asking quick questions related to this project.
A direct link is [here](https://lfaifoundation.slack.com/archives/C01J4NAL4A2).

You may also open GitHub Issues for any questions and/or suggestions you may have.

Do not use public channels to discuss any security-related issues; use instead the specific instructions provided in the [SECURITY](SECURITY.md) page.

## Contributing

We are welcoming contributions from the community.
Please consult the [CONTRIBUTING](CONTRIBUTING.md) page for help on how to proceed.
Documentation is provided in the `docs` sub-directory; the [DocumentList](docs/DocumentList.md) page provides an organized list of documents.

## Code of Conduct

The ONNX-MLIR code of conduct is described at https://onnx.ai/codeofconduct.html.
