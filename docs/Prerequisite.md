<!--- SPDX-License-Identifier: Apache-2.0 -->

# Getting the prerequisite software

<!-- Keep list below in sync with README.md. -->
```
python >= 3.8
gcc >= 6.4
protobuf >= 4.21.12
cmake >= 3.13.4
make >= 4.2.1 or ninja >= 1.10.2
java >= 1.11 (optional)
```

Onnx-mlir is tested to work with python 3.8 and 3.9 but not yet fully tested with 3.10+. It may work with older 3.x python versions but not recommended since those either have already reached or are close to reach their EOL. To check the python version, run `python --version`.

GCC can be installed with distro specific package manager on Linux such as yum on RHEL/Fedora, apt on Debian/Ubuntu, or brew on MacOS. If you prefer to compile gcc yourself, instructions can be found [here](https://gcc.gnu.org/install/). To check the gcc version, run `gcc --version`.

Protobuf can be installed with brew on MacOS. Prebuilt protobuf packages on most Linux distros do not meet the required level. You can download its binary releases or compile it yourself. The instructions can be found [here](https://github.com/protocolbuffers/protobuf). To check the protobuf version, run `protoc --version`.

Cmake can be installed with distro specific package manager on Linux such as yum on RHEL/Fedora, apt on Debian/Ubuntu, or brew on MacOS. If you prefer to compile cmake yoursellf, instructions can be found [here](https://cmake.org/install/). However, to use Cmake, you need to follow the "How to Install For Command Line Use" tutorial, which can be found in Cmake under Tools>How to Install For Command Line Use. To check the cmake version, you can either look in the desktop version under CMake>About, or run `cmake --version`.

GNU make can be installed with distro specific package manager on Linux such as yum on RHEL/Fedora, apt on Debian/Ubuntu, or brew on MacOS. If you prefer to compile make yourself, instructions can be found [here](http://git.savannah.gnu.org/cgit/make.git/tree/README.git). To check the make version, run `make --version`.

Ninja can be installed with apt on Debian/Ubuntu Linux, or brew on MacOS. On RHEL/Fedora Linux, or if you want to compile ninja yourself, the instructions can be found [here](https://ninja-build.org/). To check the ninja version, run `ninja --version`.

Java SDK can be installed with distro specific package manager on Linux such as yum on RHEL/Fedora, apt on Debian/Ubuntu, or brew on MacOS. Java SDK is only required if you plan to use the onnx-mlir `--EmitJNI` option to compile a model into a jar file for use in a Java environment. Note that the jar file contains native model runtime library called through JNI so it is not portable across different architectures. To check the java version, run `java --version`.

All the `PyPi` package dependencies and their appropriate versions are captured in [requirements.txt](requirements.txt).
