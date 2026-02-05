<!--- SPDX-License-Identifier: Apache-2.0 -->

# Outlines
This document describes how to use onnx-mlir compiler to compile and run a torch model. 

1. [Installation](#installation)
2. [Used with torch.compile](#backend)
5. [How to use onnx-mlir container inside another container](#containers)

# Installation <a name="installation"></a>
There two ways to install torch_onnxmlir package and its dependency: build and install locally, or from the pip repository for limited system configuration.

## Build and install locally
You may use the following script to build and install torch_onnxmlir package locally. If onnx-mlir source code already exists locally, the step of git clone can be skipped. The script uses directory `build-light` to avoid conflict with usually used `build` directory.
```bash
git clone --recursive https://github.com/onnx/onnx-mlir.git
cd onnx-mlir
mkdir build-light
cd build-light
cmake -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
      -DONNX_MLIR_ENABLE_PYRUNTIME_LIGHT=ON \
      ..
make OMCreateTorchONNXMLIRPackage
pip3 install -e src/Runtime/python/torch_onnxmlir
```

## Install from pip repository
Not supported yet.

# Used with torch.compile <a name="backend"></a>
In this approach, a backend based on onnxmlir compiler is provided to torch.compile(). This backend will first export the torch model to onnx model, then compile the model with onnxmlir compiler to a shared library (.so), and finally run inference with the shared library.
An example of code piece: 
```python
import torch_onnxmlir
my_option = {
    "compile_options": "-O3",
}

opt_mod = torch.compile(mod, backend="onnxmlir", options=my_option)
```

Several complete test examples are provided in [torch_onnxmlir/tests](https://github.com/onnx/onnx-mlir/blob/main/src/Runtime/python/torch_onnxmlir/tests).

## Caching the exported model and compiled library

To avoid recompling models, the backend caches compiled models in the folder `${HOME}/.cache`. Users can change the cache folder by setting an environment variable, i.e, `TORCHONNXMLIR_CACHE_DIR=path_to_cache_folder`.

# Use compiler container inside another container <a name="containers"></a>
You may run your torch env with a container. When you want to use onnx-mlir to compile the model, you need run the compiler container inside your torch container. The way to do this is to mount your directories correctly.
1. No matter whether you execute "docker run" inside a container or not, the source directory has to be the path on the host. To keep code simpile, I always keep the source and destination directory in the mount parameter the same. Otherwise, a mapping has to be passed into a docker run to convert the path when a docker run command is executed inside a container.
2. Mount all the necessary directories: the docker configuration path, and the system temporary directory, as well as your working directory. 

Here is my script to start my torch container:
```bash
DOCKER_IMAGE=ubuntu_pytorch:work

docker run -it --rm\
  -v /home1/chentong/Projects:/home1/chentong/Projects:z \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/bin/docker \
  -v /tmp:/tmp:z\
  --entrypoint '/usr/bin/bash' ${DOCKER_IMAGE}
```
Inside the iteractive docker run, the previous script can be used to build and install the torch_onnxmlir package.

Then a test case, e.g. mytest.py,  can be run with command `python mytest.py`




