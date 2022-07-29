<!--- SPDX-License-Identifier: Apache-2.0 -->

# Getting the prerequisite software

<!-- Keep list below in sync with README.md. -->
```
gcc >= 6.4
protobuf >= 3.16.0
cmake >= 3.13.4
ninja >= 1.10.2
```

GCC can be found [here](https://gcc.gnu.org/install/), or if you have [Homebrew](https://docs.brew.sh/Installation), you can use `brew install gcc`. To check what version of gcc you have installed, run `gcc --version`.

The instructions to install libprotoc can be found [here](https://google.github.io/proto-lens/installing-protoc.html). Or alternatively, if you have Homebrew, you can run `brew install protobuf`. To check what version you have installed, run `protoc --version`. 
Custom directions for installing protobuf under Windows are provided [here](BuildOnWindows.md#protobuf).

Cmake can be found [here](https://cmake.org/download/). However, to use Cmake, you need to follow the "How to Install For Command Line Use" tutorial, which can be found in Cmake under Tools>How to Install For Command Line Use. To check which version you have, you can either look in the desktop version under CMake>About, or run `cmake --version`.

The instructions for installing Ninja can be found [here](https://ninja-build.org/). Or, using Homebrew, you can run `brew install ninja`. To check the version, run `ninja --version`.
