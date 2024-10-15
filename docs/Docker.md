<!--- SPDX-License-Identifier: Apache-2.0 -->

# Building and Developping ONNX-MLIR using Docker

There are three ways to use ONNX-MLIR with Docker.
1. [Using a prebuild image](#prebuilt-containers), recommended for using ONNX-MLIR but not developing it.
2. [Using a script](#easy-script-to-compile-a-model), recommended for testing our infrastructure quickly without explicitly installing a Docker image.
3. [Using a custom build image](#building-and-developping-onnx-mlir-using-docker), recommended for developing ONNX-MLIR.

## Prebuilt Images

An easy way to get started with ONNX-MLIR is to use a prebuilt Docker image.
These images are created as a result of a successful merge build on the trunk.
This means that the latest image represents the tip of the trunk.
Currently there are both Release and Debug mode images for `amd64`, `ppc64le` and `s390x` saved in Docker Hub as, respectively, [onnxmlir/onnx-mlir](https://github.com/users/onnxmlir/packages/container/onnx-mlir) and [onnxmlir/onnx-mlir-dev](https://github.com/users/onnxmlir/packages/container/onnx-mlir-dev).
To use one of these images either pull it directly from Docker Hub, launch a container and run an interactive bash shell in it, or use it as the base image in a Dockerfile.

Here are the differences between the two Docker images.
* The `onnx-mlir` image just contains the built compiler and you can use it immediately to compile your model without any installation. It does not include any support to run compiled model.
* The `onnx-mlir-dev` image contains the built compiler plus all of the tools and support needed for development, including support to run our tests locally. The image also support tools to run compiled models, such as support for our python interface.

## Easy Script to Compile a Model

A python convenience script is provided to allow you to run ONNX-MLIR inside a Docker container as if running the ONNX-MLIR compiler directly on the host.
The resulting output is an Linux ELF library implementing the ONNX model.
The `onnx-mlir.py` script is located in the [docker](../docker) directory. For example, compiling a MNIST model can be done as follows.
```
# docker/onnx-mlir.py -O3 --EmitLib mnist/model.onnx
505a5a6fb7d0: Pulling fs layer
505a5a6fb7d0: Verifying Checksum
505a5a6fb7d0: Download complete
505a5a6fb7d0: Pull complete
Shared library model.so has been compiled.
```

The script will pull the onnx-mlir image if it's not available locally, mount the directory containing the `model.onnx` into the container, and compile and generate the `model.so` in the same directory.

This script takes the same option as the normal  `onnx-mlir` command used to compile a ONNX model. Typical options are `-O0` (default) or `-O3` to define an optimization level and `--EmitLib` (default) or `--EmitJNI` to generate a dynamic library or a jar file.
A complete list of options is provided by using the traditional `--help` option.

This script generates codes that can be executed on a Linux system or within a Docker container.

## Building ONNX-MLIR in a docker environment

The onnx-mlir-dev image contains the full build tree including the prerequisites and a clone of the source code.
The source can be modified and `onnx-mlir` can be rebuilt from within the container, so it is possible to use it as a development environment.
New pull requests can be generated, and the repository can be updated to the latest using git commands.
It is also possible to attach vscode to the running container.
An example Dockerfile useful for development and vscode configuration files can be seen in the [docs/docker-example](docker-example) folder.
If the workspace directory and the vscode files are not present in the directory where the Docker build is run, then the lines referencing them should be commented out or deleted.

The Dockerfile is shown here, and should be modified according to one's need. The file below includes debugging tools as well as pytorch, which can be used to train the mnist model in our end-to-end example provided in the [docs/mnist_example](mnist_example) directory.

[same-as-file]: <> (docs/docker-example/Dockerfile)
```
FROM ghcr.io/onnxmlir/onnx-mlir-dev
WORKDIR /workdir
ENV HOME=/workdir

# 1) Install packages.
ENV PATH=$PATH:/workdir/bin
RUN apt-get update
RUN apt-get install -y python3-numpy
RUN apt-get install -y python3-pip
RUN python -m pip install --upgrade pip
RUN apt-get install -y gdb
RUN apt-get install -y lldb

# 2) Instal optional packages, comment/uncomment/add as you see fit.
RUN apt-get install -y vim
RUN apt-get install -y emacs
RUN apt-get install -y valgrind
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y clang-format
RUN python -m pip install wheel
RUN python -m pip install numpy
RUN python -m pip install torch==2.0.0+cpu torchvision==0.15.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN git clone https://github.com/onnx/tutorials.git
# Install clang
RUN apt-get install -y lsb-release wget software-properties-common
RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
# For development
RUN apt-get install -y ssh-client

# 3) When using vscode, copy your .vscode in the Dockerfile dir and
#    uncomment the two lines below.
# WORKDIR /workdir/.vscode
# ADD .vscode /workdir/.vscode

# 4) When using a personal workspace folder, set your workspace sub-directory
#    in the Dockerfile dir and uncomment the two lines below.
# WORKDIR /workdir/workspace
# ADD workspace /workdir/workspace

# 5) Fix git by reattaching head and making git see other branches than main.
WORKDIR /workdir/onnx-mlir
# Add optional personal fork and disable pushing to upstream (best practice).
# RUN git remote add origin https://github.com/<<user>>/onnx-mlir.git
# RUN git remote set-url --push upstream no_push

# 6) Set the PATH environment vars for make/debug mode. Replace Debug
#    with Release in the PATH below when using Release mode.
WORKDIR /workdir
ENV NPROC=4
ENV PATH=$PATH:/workdir/onnx-mlir/build/Debug/bin/:/workdir/onnx-mlir/build/Debug/lib:/workdir/llvm-project/build/bin
```

The first step is to copy the [docs/docker-example](docker-example) directory to another directory outside of the repo, say `~/DockerOnnxMlir`. Or simply download the `Dockerfile` and the `.vscode` file if you intend to use VSCode.

Then, the `Dockerfile` in the copied directory should then be modified to suit one's need. In particular, we recommend developers to use their own fork for development. Uncomment the lines associated with git (Step 5 in the file) and substitute the appropriate GitHub Id in the commented out directives. 
The lines associated with VSCode (Step 3 in the file) should be also uncommented when using VSCode. 
Finally, we recommend creating a subdirectory named `workspace` that contains test examples you would like to have in your Docker Image and Container. 
If so, uncomment the lines associated with copying a personal workspace folder (Step 4 in the file), and that subdirectory's content will be copied over to the Docker Image.

The next step is to create a Docker image. This step can be performed using the `docker build --tag imageName .` shell command. Once this command is successful, we must start a container. This can be done by a command line (e.g. `docker run -it imageName`) or by opening the Docker Dashboard, locating the Image Tab, and clicking the `run` button associated with the image just created (e.g. `imageName` above).

These steps are summarized here.
``` shell
# Starting in the onnx-mlir directory, copy the Docker example directory.
cp -prf docs/docker-example ~/DockerOnnxMlir
cd ~/DockerOnnxMlir
# Edit the Dockerfile.
vi Dockerfile
# Build the Docker image.
docker build --tag ghcr.io/onnxmlir/onnx-mlir-dev .
# Start a container using the Docker dashboard or a docker run command.
docker run -it ghcr.io/onnxmlir/onnx-mlir-dev
```

**NOTE:** If you are using a MacBook with the Apple M1 chip, please follow the steps below for configuration:
``` shell
# Starting in the onnx-mlir directory, copy the Docker example directory.
cp -prf docs/docker-example ~/DockerOnnxMlir
cd ~/DockerOnnxMlir
# Edit the Dockerfile.
vi Dockerfile
# Pull the Docker image with the specified platform
docker pull --platform linux/amd64 ghcr.io/onnxmlir/onnx-mlir-dev
# Build the Docker image.
docker build --platform linux/amd64 --tag ghcr.io/onnxmlir/onnx-mlir-dev .
# Start a container using the Docker dashboard or a docker run command.
docker run --platform linux/amd64 -it ghcr.io/onnxmlir/onnx-mlir-dev
```

Tip: Instead of adding the platform flag for every docker pull, build, and run command. You can set the environment variable `DOCKER_DEFAULT_PLATFORM` and use the first set of steps:
```
export DOCKER_DEFAULT_PLATFORM=linux/amd64
```

### Developing with Docker in VSCode

The next step is to open VSCode, load the Docker Extension if not already present, and then open the Docker tab on the left pane. Locate the container that was just started in the previous step, right click on it, and select the `Attach Visual Studio Code` option.
This will open a new VSCode window. Open a local folder on the `workdir` directory, this will give you access to all of the ONNX/MLIR/LLVM code as well as the `workspace` subdirectory.

You may then open a shell, go to the `onnx-mlir` subdirectory, and check that all of the git is properly setup.

If you opted to add your own fork, it will be listed under `origin` with `upstream` being the official ONNX-MLIR repo. For example:
``` shell
git remote -v
#origin   https://github.com/AlexandreEichenberger/onnx-mlir.git (fetch)
#origin   https://github.com/AlexandreEichenberger/onnx-mlir.git (push)
#upstream https://github.com/onnx/onnx-mlir.git (fetch)
#upstream no_push (push)
```

Now, you may fetch your own branches using `git fetch origin`, and switch to one of your branch (say `my-opt`) using the `git checkout --track origin/my-opt` command. The `--track` option is recommended as `upstream` was cloned and `origin` was added as remote. Once you want to push your changes, you should use `git push -u origin my-opt`, using the `-u` option to link the local branch with the `origin` remote repo.

The `main` branch will default to the upstream repo. If you prefer it to be associated with your own fork's `main` branch, you may update your main branch to the latest and associate the local main branch with `origin` using the commands listed below.
``` shell
git checkout main
git branch --unset-upstream
git push --set-upstream origin main
```

A Docker container can be used to investigate a bug, or to develop a new feature. Some like to create a new images for each new version of ONNX-MLIR; others prefer to create one image and use git to update the main branch and use git to switch between multiple branches. Both are valid approaches.

## Using a devcontainer
Another way of building onnx-mlir for development in VSCode is using a devcontainer. This way you only mount your source folder, meaning that changes you do are saved on your local machine. For this setup to work you need a `Dockerfile` and a `devcontainer.json` file. Both are provided in `docs/devcontainer-example`. 

The [`Dockerfile`](devcontainer-example/Dockerfile.llvm-project) is a simple Dockerfile based on the precompiled LLVM/MLIR image that is shared. It installs additional software that is useful for developing and also sets `LLVM_PROJECT_ROOT` to easily refer to the LLVM path.


The [`devcontainer.json`](devcontainer-example/devcontainer.json) preinstalls extensions and defines settings for the VS Code server running inside the container. This way you don't have to setup VS Code everytime you enter the container. In `postAttachCommand` ONNX is installed.

To use this setup you first clone onnx-mlir and all submodules (for example with` git clone --recursive https://github.com/onnx/onnx-mlir.git`). You then create a new folder named `.devcontainer` in the source root. After that you copy the two files in `docs/devcontainer-example` into that folder. Now simply press `CTRL+SHIFT+P` and execute `Dev Containers: Reopen in Container`. VSCode will now create the docker image and mount the source folder.

You can now configure onnx-mlir as described in [BuildOnLinuxOSX](BuildOnLinuxOSX.md). `MLIR_DIR` is already set for you, so you can skip that step.

**Note:** To run this on M1/2 Macs something like Rosetta is needed. This is related to https://github.com/docker/roadmap/issues/384
