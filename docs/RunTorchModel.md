<!--- SPDX-License-Identifier: Apache-2.0 -->

# Outlines
This document describes how to use onnx-mlir compiler to compile and run a torch model. 

1. [Installation](#installation)
2. [Used with torch.compile](#backend)
3. [Used as a model wrapper](#wrapper)
4. [Used for debugging](#debug) 
5. [How to use onnx-mlir container inside another container](#containers)

# Installation <a name="installation"></a>
There two ways to install onnxmlirtorch package and its dependency: build and install locally, or from the pip repository for limited system configuration.

## Build and install locally
You may use the following script to build and install onnxmlirtorch package locally. If onnx-mlir source code already exists locally, the step of git clone can be skipped. The script uses directory `build-light` to avoid conflict with usually used `build` directory.
```bash
git clone --recursive https://github.com/onnx/onnx-mlir.git
cd onnx-mlir
mkdir build-light
cd build-light
cmake -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
      -DONNX_MLIR_ENABLE_PYRUNTIME_LIGHT=ON \
      ..
make OMCreateONNXMLIRTorchPackage
pip3 install -e src/Runtime/python/onnxmlirtorch
```
## Install from pip repository
Not supported yet.

# Used with torch.compile <a name="backend"></a>
In this approach, a backend based on onnxmlir compiler is provided to torch.compile(). This backend will first export the torch model to onnx model, then compile the model with onnxmlir compiler to a shared library (.so), and finally run inference with the shared library.
An example of code piece: 
```
my_option = {
    "compile_options": "-O3",
}

opt_mod = torch.compile(mod, backend=onnxmlirtorch.onnxmlir_backend, options=my_option)
```

Several complete test examples are provided in [onnxmlirtorch/tests](https://github.com/onnx/onnx-mlir/blob/main/src/Runtime/python/onnxmlirtorch/tests):
. torch_compile_container_add.py: The default container for onnx-mlir compiler is used to compile the model
. torch_compile_container_nnpa.py: With the default container for onnx-mlir compiler, the compilation flag for z16 NNPA is specified.
. torch_compile_add.py: Compile with  a locally available compiler. You need to modify the compiler path to your own if you want to try.


# Used as a model Wrapper <a name="wrapper"></a>
In this approach, the torch model is wrapped with a new class, the forward() function of which is to export, compile and run the model with onnx-mlir compiler. Different from the torch.compile approach, the model will NOT be optimized/changed. It's simpler than using torch.compile, but only supports the forward function, not other possible member functions or values.
An example, torch_wrap_add.py can be found in the onnxlirtorch/tests.
```
import onnxmlirtorch

...

opt_mod = onnxmlirtorch.compile(mod, **my_option)
```
Or in another way:

```
import onnxmlirtorch

...

opt_mod = onnxmlirtorch.ONNXMLIRTorch(mod, **my_option)
results = opt_mod(inputs)
```

## Caching the exported model and compiled library
If the model remains the same for two different inputs, we can reuse the execution session already generated and avoid the overhead of exporting and compiling onnx models. In the wrap approach, a session cache is introduced. The key for the cache is the shape of all the inputs.
In the torch.compiler(), the model may change even if the shape of inputs for two inference are  the same. No cache is implemented for torch.compile so far. Cache mechanism may be added when the comparison of models is added.
The conflict in file name and entry name in dynamic library has to be avoided, xwhen mulitple dynamic libraries are created. "Tag" from a global counter is attached to each file name or entry name.
The cache is implemented with sessioncache.py. LRU replacement policy is used. The default cache size is 3.
The current implementation is not using any dynamic dimension in model exporting.

# Used for debugging <a name="debug"></a>
This is a side product of this package. A function to print the inputs is provided and hooked with torch.nn.modules.module.register_module_forward_hook.
Example:
```
onnxmlirtorch.interceptForward(model)
```
Again the input parameters will be printed out after the forward() function is called. You may add your code in function print_parameter() in onnxmlirtorch.py

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
Inside the iteractive docker run, the previous script can be used to build and install the onnxmlirtorch package.

Then a test case, e.g. mytest.py,  can be run with command `python mytest.py`




